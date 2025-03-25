/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "candidates.hpp"
#include "device.hpp"
#include "graph.hpp"
#include "pool.hpp"
#include "signature.hpp"
#include "types.hpp"
#include "utils.hpp"
#include <sycl/sycl.hpp>


namespace sigmo {
namespace isomorphism {
namespace filter {

template<CandidatesDomain D = CandidatesDomain::Query>
utils::BatchedEvent filterCandidates(sycl::queue& queue,
                                     sigmo::DeviceBatchedCSRGraph& query_graph,
                                     sigmo::DeviceBatchedCSRGraph& data_graph,
                                     sigmo::signature::Signature<>& signatures,
                                     sigmo::candidates::Candidates& candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;

  sycl::range<1> local_range{device::deviceOptions.filter_work_group_size};
  sycl::range<1> global_range{total_data_nodes + (local_range[0] - (total_data_nodes % local_range[0]))};

  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<sigmo::device::kernels::FilterCandidatesKernel<D>>(sycl::nd_range<1>({global_range, local_range}),
                                                                        [=,
                                                                         candidates = candidates.getCandidatesDevice(),
                                                                         query_signatures = signatures.getDeviceQuerySignatures(),
                                                                         data_signatures = signatures.getDeviceDataSignatures(),
                                                                         max_labels = signatures.getMaxLabels()](sycl::nd_item<1> item) {
                                                                          auto data_node_id = item.get_global_id(0);
                                                                          if (data_node_id >= total_data_nodes) { return; }
                                                                          auto data_signature = data_signatures[data_node_id];
                                                                          auto query_labels = query_graph.labels;
                                                                          auto data_labels = data_graph.labels;

                                                                          for (size_t query_node_id = 0; query_node_id < total_query_nodes;
                                                                               ++query_node_id) {
                                                                            if (query_labels[query_node_id] != data_labels[data_node_id]) {
                                                                              continue;
                                                                            }
                                                                            if constexpr (D == CandidatesDomain::Data) {
                                                                              candidates.insert(data_node_id, query_node_id);
                                                                            } else {
                                                                              candidates.atomicInsert(query_node_id, data_node_id);
                                                                            }
                                                                          }
                                                                        });
  });

  utils::BatchedEvent be;
  be.add(e);
  return be;
}

template<CandidatesDomain D = CandidatesDomain::Query>
utils::BatchedEvent refineCandidates(sycl::queue& queue,
                                     sigmo::DeviceBatchedCSRGraph& query_graph,
                                     sigmo::DeviceBatchedCSRGraph& data_graph,
                                     sigmo::signature::Signature<>& signatures,
                                     sigmo::candidates::Candidates& candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;

  sycl::range<1> local_range{device::deviceOptions.filter_work_group_size};
  sycl::range<1> global_range{total_data_nodes + (local_range[0] - (total_data_nodes % local_range[0]))};

  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<sigmo::device::kernels::RefineCandidatesKernel<D>>(
        sycl::nd_range<1>({global_range, local_range}),
        [=,
         candidates = candidates.getCandidatesDevice(),
         query_signatures = signatures.getDeviceQuerySignatures(),
         data_signatures = signatures.getDeviceDataSignatures(),
         max_labels = signatures.getMaxLabels()](sycl::nd_item<1> item) {
          auto data_node_id = item.get_global_id(0);
          if (data_node_id >= total_data_nodes) { return; }
          auto data_signature = data_signatures[data_node_id];

          for (size_t query_node_id = 0; query_node_id < total_query_nodes; ++query_node_id) {
            if constexpr (D == CandidatesDomain::Data) {
              if (!candidates.contains(data_node_id, query_node_id)) { continue; }
            } else {
              if (!candidates.atomicContains(query_node_id, data_node_id)) { continue; }
            }
            auto query_signature = query_signatures[query_node_id];

            bool keep = true;
            for (types::label_t l = 0; l < max_labels && keep; l++) {
              keep = keep && (query_signature.getLabelCount(l) <= data_signature.getLabelCount(l));
            }
            if (!keep) {
              if constexpr (D == CandidatesDomain::Data) {
                candidates.remove(data_node_id, query_node_id);
              } else {
                candidates.atomicRemove(query_node_id, data_node_id);
              }
            }
          }
        });
  });

  utils::BatchedEvent be;
  be.add(e);
  return be;
}

} // namespace filter

namespace join {
/**
 * // TODO improve data locality and shared memory usage
 * // TODO design data structure for temporary store the candidates for each query graph in shared memory
 * // TODO improve readibility
 */

struct Stack {
  uint depth;
  size_t candidateIdx;
};

struct Mapping { // TODO: make it SOA
  size_t query_graph_id;
  size_t data_graph_id;
  types::node_t query_nodes[10];
  types::node_t data_nodes[10];
};

SYCL_EXTERNAL bool isValidMapping(types::node_t candidate,
                                  uint depth,
                                  const uint32_t* mapping,
                                  const sigmo::DeviceBatchedCSRGraph& query_graphs,
                                  uint query_graph_id,
                                  const sigmo::DeviceBatchedCSRGraph& data_graphs,
                                  uint data_graph_id) {
  for (int i = 0; i < depth; i++) {
    size_t query_nodes_offset = query_graphs.getPreviousNodes(query_graph_id);

    if (query_graphs.isNeighbor(i + query_nodes_offset, depth + query_nodes_offset) != data_graphs.isNeighbor(mapping[i], candidate)) {
      return false;
    }
  }

  return true;
}

SYCL_EXTERNAL void defineMatchingOrder(sycl::sub_group sg, size_t num_query_nodes, types::node_t* matching_order, size_t& starting_node_candidates) {
  size_t tmp = starting_node_candidates;

  int starting_node = sg.get_local_linear_id();

  starting_node_candidates
      = sycl::reduce_over_group(sg, starting_node_candidates, sycl::maximum<>()); // the starting node is the one with the most candidates
  if (starting_node >= num_query_nodes || starting_node_candidates != tmp) { starting_node = -1; }
  starting_node = sycl::reduce_over_group(sg, starting_node, sycl::maximum<>());
  matching_order[0] = starting_node;

  for (int i = 0, j = 1; i < num_query_nodes; i++) {
    if (i == starting_node) { continue; }
    matching_order[j++] = i;
  }
}

utils::BatchedEvent joinCandidates2(sycl::queue& queue,
                                    sigmo::DeviceBatchedCSRGraph& query_graphs,
                                    sigmo::DeviceBatchedCSRGraph& data_graphs,
                                    sigmo::candidates::Candidates& candidates,
                                    sigmo::isomorphism::mapping::GMCR& gmcr,
                                    size_t* num_matches,
                                    bool find_first = true) {
  utils::BatchedEvent e;
  const size_t total_query_nodes = query_graphs.total_nodes;
  const size_t total_data_nodes = data_graphs.total_nodes;
  const size_t total_query_graphs = query_graphs.num_graphs;
  const size_t total_data_graphs = data_graphs.num_graphs;

  const size_t preferred_workgroup_size = device::deviceOptions.join_work_group_size;

  size_t size = gmcr.getGMCRDevice().data_graph_offsets[total_data_graphs];

  size = size + (preferred_workgroup_size - (size % preferred_workgroup_size));

  sycl::nd_range<1> nd_range{size, preferred_workgroup_size};
  constexpr size_t MAX_QUERY_NODES = 30;
  auto e1 = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<device::kernels::JoinCandidates2Kernel>(
        nd_range,
        [=, query_graphs = query_graphs, data_graphs = data_graphs, candidates = candidates.getCandidatesDevice(), gmcr = gmcr.getGMCRDevice()](
            sycl::nd_item<1> item) {
          const size_t wgid = item.get_group(0);
          if (wgid >= size) { return; }
          const uint32_t query_graph_id = gmcr.query_graph_indices[wgid];

          sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> num_matches_ref{num_matches[0]};

          types::node_t mapping[MAX_QUERY_NODES];
          size_t private_num_matches = 0;

          size_t data_graph_id = utils::binarySearch(gmcr.data_graph_offsets, total_data_graphs, query_graph_id);

          const uint32_t start_data_graph = data_graphs.graph_offsets[data_graph_id];
          const uint32_t end_data_graph = data_graphs.graph_offsets[data_graph_id + 1];

          Stack stack[MAX_QUERY_NODES]; // TODO: assume max depth of 30 but make it dynamic
          const uint32_t offset_query_nodes = query_graphs.getPreviousNodes(query_graph_id);
          const uint16_t num_query_nodes = query_graphs.getGraphNodes(query_graph_id);

          // start DFS
          utils::detail::Bitset<uint64_t> visited{start_data_graph};
          uint top = 0;
          stack[top++] = {0, 0}; // initialize stack with the first node
          // DFS loop
          while (top > 0) {
            // get the top frame
            auto frame = stack[top - 1];
            auto query_node = frame.depth;

            if (frame.depth == num_query_nodes) { // found a match and output solution
              private_num_matches++;
              top--;
              if (find_first) {
                break;
              } else {
                continue;
              }
            }
            // no more candidates
            if (frame.candidateIdx >= candidates.getCandidatesCount(query_node + offset_query_nodes, start_data_graph, end_data_graph)) {
              // backtrack
              top--;
              // free the failed mapping
              visited.unset(mapping[query_node]);
              // clear visited if we are back to the first node
              if (top == 1) { visited.clear(); }
              continue;
            }

            // try the next candidate
            auto candidate = candidates.getCandidateAt(query_node + offset_query_nodes, frame.candidateIdx, start_data_graph, end_data_graph);

            // increment the candidate index for the next iteration
            stack[top - 1].candidateIdx++;

            // if the candidate is already in the mapping, skip it
            if (visited.get(candidate)) { continue; }

            // check if the candidate is valid
            if ((frame.depth == 0 || isValidMapping(candidate, frame.depth, mapping, query_graphs, query_graph_id, data_graphs, data_graph_id))) {
              mapping[frame.depth] = candidate;
              visited.set(candidate);
              stack[top++] = {frame.depth + 1, 0};
            }
          }


          num_matches_ref += private_num_matches;
        });
  });

  e.add(e1);
  return e;
}

utils::BatchedEvent joinCandidates(sycl::queue& queue,
                                   sigmo::DeviceBatchedCSRGraph& query_graphs,
                                   sigmo::DeviceBatchedCSRGraph& data_graphs,
                                   sigmo::candidates::Candidates& candidates,
                                   sigmo::isomorphism::mapping::GMCR& gmcr,
                                   size_t* num_matches,
                                   bool find_first = true) {
  utils::BatchedEvent e;
  const size_t total_query_nodes = query_graphs.total_nodes;
  const size_t total_data_nodes = data_graphs.total_nodes;
  const size_t total_query_graphs = query_graphs.num_graphs;
  const size_t total_data_graphs = data_graphs.num_graphs;

  const size_t preferred_workgroup_size = device::deviceOptions.join_work_group_size;

  sycl::nd_range<1> nd_range{total_data_graphs * preferred_workgroup_size, preferred_workgroup_size};
  constexpr size_t MAX_QUERY_NODES = 30;
  auto e1 = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<device::kernels::JoinCandidatesKernel>(
        nd_range,
        [=, query_graphs = query_graphs, data_graphs = data_graphs, candidates = candidates.getCandidatesDevice(), gmcr = gmcr.getGMCRDevice()](
            sycl::nd_item<1> item) {
          const size_t lid = item.get_local_linear_id();
          const size_t gid = item.get_global_linear_id();

          const auto wg = item.get_group();
          const size_t wgid = wg.get_group_linear_id();
          const size_t wglid = wg.get_local_linear_id();
          const size_t wgsize = wg.get_local_range()[0];

          const auto sg = item.get_sub_group();
          const size_t sgid = sg.get_group_linear_id();
          const size_t sglid = sg.get_local_linear_id();
          const size_t sgsize = sg.get_local_range()[0];

          sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> num_matches_ref{num_matches[0]};

          types::node_t mapping[MAX_QUERY_NODES];
          size_t private_num_matches = 0;

          for (uint32_t data_graph_id = wgid; data_graph_id < total_data_graphs; data_graph_id += wg.get_group_linear_range()) {
            const uint32_t start_data_graph = data_graphs.graph_offsets[data_graph_id];
            const uint32_t end_data_graph = data_graphs.graph_offsets[data_graph_id + 1];

            const uint32_t start_query = gmcr.data_graph_offsets[data_graph_id];
            const uint32_t end_query = gmcr.data_graph_offsets[data_graph_id + 1];

            for (uint32_t query_graph_it = wglid; query_graph_it < (end_query - start_query);
                 query_graph_it += wgsize) { // iterate over all query graphs
              const uint32_t query_graph_id = gmcr.query_graph_indices[start_query + query_graph_it];
              Stack stack[MAX_QUERY_NODES]; // TODO: assume max depth of 30 but make it dynamic
              const uint32_t offset_query_nodes = query_graphs.getPreviousNodes(query_graph_id);
              const uint16_t num_query_nodes = query_graphs.getGraphNodes(query_graph_id);

              // start DFS
              utils::detail::Bitset<uint64_t> visited{start_data_graph};
              uint top = 0;
              stack[top++] = {0, 0}; // initialize stack with the first node
              // DFS loop
              while (top > 0) {
                // get the top frame
                auto frame = stack[top - 1];
                auto query_node = frame.depth;

                if (frame.depth == num_query_nodes) { // found a match and output solution
                  private_num_matches++;
                  top--;
                  if (find_first) {
                    break;
                  } else {
                    continue;
                  }
                }
                // no more candidates
                if (frame.candidateIdx >= candidates.getCandidatesCount(query_node + offset_query_nodes, start_data_graph, end_data_graph)) {
                  // backtrack
                  top--;
                  // free the failed mapping
                  visited.unset(mapping[query_node]);
                  // clear visited if we are back to the first node
                  if (top == 1) { visited.clear(); }
                  continue;
                }

                // try the next candidate
                auto candidate = candidates.getCandidateAt(query_node + offset_query_nodes, frame.candidateIdx, start_data_graph, end_data_graph);

                // increment the candidate index for the next iteration
                stack[top - 1].candidateIdx++;

                // if the candidate is already in the mapping, skip it
                if (visited.get(candidate)) { continue; }

                // check if the candidate is valid
                if ((frame.depth == 0 || isValidMapping(candidate, frame.depth, mapping, query_graphs, query_graph_id, data_graphs, wgid))) {
                  mapping[frame.depth] = candidate;
                  visited.set(candidate);
                  stack[top++] = {frame.depth + 1, 0};
                }
              }
            }
          }
          num_matches_ref += private_num_matches;
        });
  });

  e.add(e1);
  return e;
}

} // namespace join
} // namespace isomorphism
} // namespace sigmo