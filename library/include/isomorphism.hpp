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
#include <sycl/sycl.hpp>


namespace mbsm {
namespace isomorphism {
namespace filter {

template<candidates::CandidatesDomain D = candidates::CandidatesDomain::Query>
utils::BatchedEvent filterCandidates(sycl::queue& queue,
                                     mbsm::DeviceBatchedQueryGraph& query_graph,
                                     mbsm::DeviceBatchedDataGraph& data_graph,
                                     mbsm::signature::Signature<>& signatures,
                                     mbsm::candidates::Candidates& candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;
  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<mbsm::device::kernels::FilterCandidatesKernel<D>>(
        sycl::range<1>(total_data_nodes),
        [=,
         candidates = candidates.getCandidatesDevice(),
         query_signatures = signatures.getDeviceQuerySignatures(),
         data_signatures = signatures.getDeviceDataSignatures(),
         max_labels = signatures.getMaxLabels()](sycl::item<1> item) {
          auto data_node_id = item.get_id(0);
          auto data_signature = data_signatures[data_node_id];
          auto query_labels = query_graph.labels;
          auto data_labels = data_graph.labels;

          for (size_t query_node_id = 0; query_node_id < total_query_nodes; ++query_node_id) {
            if (query_labels[query_node_id] != data_labels[data_node_id]) { continue; }
            auto query_signature = query_signatures[query_node_id];

            bool insert = true;
            for (types::label_t l = 0; l < max_labels; l++) {
              insert = insert && (query_signature.getLabelCount(l) <= data_signature.getLabelCount(l));
              if (!insert) break;
            }
            if (insert) {
              if constexpr (D == candidates::CandidatesDomain::Data) {
                candidates.insert(data_node_id, query_node_id);
              } else {
                candidates.atomicInsert(query_node_id, data_node_id);
              }
            }
          }
        });
  });

  utils::BatchedEvent be;
  be.add(e);
  return be;
}

template<candidates::CandidatesDomain D = candidates::CandidatesDomain::Query>
utils::BatchedEvent refineCandidates(sycl::queue& queue,
                                     mbsm::DeviceBatchedQueryGraph& query_graph,
                                     mbsm::DeviceBatchedDataGraph& data_graph,
                                     mbsm::signature::Signature<>& signatures,
                                     mbsm::candidates::Candidates& candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;
  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<mbsm::device::kernels::RefineCandidatesKernel<D>>(
        sycl::range<1>(total_data_nodes),
        [=,
         candidates = candidates.getCandidatesDevice(),
         query_signatures = signatures.getDeviceQuerySignatures(),
         data_signatures = signatures.getDeviceDataSignatures(),
         max_labels = signatures.getMaxLabels()](sycl::item<1> item) {
          auto data_node_id = item.get_id(0);
          auto data_signature = data_signatures[data_node_id];

          for (size_t query_node_id = 0; query_node_id < total_query_nodes; ++query_node_id) {
            if constexpr (D == candidates::CandidatesDomain::Data) {
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
              if constexpr (D == candidates::CandidatesDomain::Data) {
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
namespace mapping {

struct DGCR {
  uint32_t* data_graph_offsets;
  uint32_t* query_graph_indices;
};


// Offloaded version of generateDGCR using SYCL kernels.
DGCR generateDGCR(sycl::queue& queue,
                  mbsm::DeviceBatchedQueryGraph& query_graphs,
                  mbsm::DeviceBatchedDataGraph& data_graphs,
                  mbsm::candidates::Candidates& candidates,
                  bool find_all = true) {
  // Get dimensions
  const size_t total_query_graphs = query_graphs.num_graphs;
  const size_t total_data_graphs = data_graphs.num_graphs;

  // Allocate device memory for data_graph_offsets (size = total_data_graphs+1)
  uint32_t* d_data_graph_offsets = sycl::malloc_shared<uint32_t>(total_data_graphs + 1, queue);
  // Initialize to zero
  queue.fill(d_data_graph_offsets, 0, total_data_graphs + 1).wait();

  // --- Kernel 1: Count query graphs (with > 1 node) per data graph ---
  // For each pair (query_graph, data_graph), if every node in the query graph
  // has a candidate in the given data graph, then atomically increment the counter.
  auto k1 = queue.parallel_for(
      sycl::range<2>(total_query_graphs, total_data_graphs),
      [=, query_graphs = query_graphs, data_graphs = data_graphs, candidates = candidates.getCandidatesDevice()](sycl::id<2> idx) {
        size_t query_graph_id = idx[0];
        size_t data_graph_id = idx[1];
        size_t num_query_nodes = query_graphs.getGraphNodes(query_graph_id);

        size_t offset_query_nodes = query_graphs.getPreviousNodes(query_graph_id);
        bool add = true;
        for (size_t i = 0; i < num_query_nodes && add; i++) {
          size_t global_query_node = offset_query_nodes + i;
          size_t start_data = data_graphs.graph_offsets[data_graph_id];
          size_t end_data = data_graphs.graph_offsets[data_graph_id + 1];
          add = add && (candidates.getCandidatesCount(global_query_node, start_data, end_data) > 0);
        }
        if (add) {
          // Increment the counter for this data graph.
          sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>
              atomic_offset(d_data_graph_offsets[data_graph_id + 1]);
          atomic_offset.fetch_add(1);
        }
      });
  k1.wait();

  // --- Kernel 2: Compute prefix sum of data_graph_offsets ---
  // (Here we assume total_data_graphs is small so a single task is acceptable.)
  for (size_t i = 1; i <= total_data_graphs; i++) { d_data_graph_offsets[i] += d_data_graph_offsets[i - 1]; }

  // The total number of query indices is now the last element in d_data_graph_offsets.
  uint32_t total_query_indices = d_data_graph_offsets[total_data_graphs];
  // Allocate device memory for query_graph_indices.
  uint32_t* d_query_graph_indices = sycl::malloc_shared<uint32_t>(total_query_indices, queue);

  // Create a copy of the prefix sum array to serve as atomic “current offsets”
  uint32_t* current_offsets = sycl::malloc_shared<uint32_t>(total_data_graphs + 1, queue);
  for (size_t i = 0; i <= total_data_graphs; i++) { current_offsets[i] = d_data_graph_offsets[i]; }

  // --- Kernel 3: Fill query_graph_indices ---
  // For each (query_graph, data_graph) pair (with query graphs having > 1 node),
  // if the graph qualifies (each query node has a candidate in the data graph),
  // then atomically obtain an index (from current_offsets) and write the query_graph_id.
  auto k3 = queue.parallel_for(
      sycl::range<2>(total_query_graphs, total_data_graphs),
      [=, query_graphs = query_graphs, data_graphs = data_graphs, candidates = candidates.getCandidatesDevice()](sycl::id<2> idx) {
        size_t query_graph_id = idx[0];
        size_t data_graph_id = idx[1];
        size_t num_query_nodes = query_graphs.getGraphNodes(query_graph_id);
        if (num_query_nodes <= 1) return; // skip
        size_t offset_query_nodes = query_graphs.getPreviousNodes(query_graph_id);
        bool add = true;
        for (size_t i = 0; i < num_query_nodes && add; i++) {
          size_t global_query_node = offset_query_nodes + i;
          size_t start_data = data_graphs.graph_offsets[data_graph_id];
          size_t end_data = data_graphs.graph_offsets[data_graph_id + 1];
          add = add && (candidates.getCandidatesCount(global_query_node, start_data, end_data) > 0);
        }
        if (add) {
          sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_off(
              current_offsets[data_graph_id]);
          uint32_t index = atomic_off.fetch_add(1);
          d_query_graph_indices[index] = static_cast<uint32_t>(query_graph_id);
        }
      });
  k3.wait();

  // Build the result structure.
  DGCR dgcr;
  dgcr.data_graph_offsets = d_data_graph_offsets;
  dgcr.query_graph_indices = d_query_graph_indices;
  sycl::free(current_offsets, queue);
  return dgcr;
}

} // namespace mapping


namespace join {
/**
 * // TODO improve data locality and shared memory usage
 * // TODO design data structure for temporary store the candidates for each query graph in shared memory
 * // TODO improve readibility
 */

struct Stack {
  int depth;
  size_t candidateIdx;
};

struct Visited {
  size_t offset;
  uint64_t visited;

  Visited(size_t offset) : offset(offset), visited(0) {}

  SYCL_EXTERNAL bool get(size_t idx) const { return visited & (static_cast<uint64_t>(1) << (idx - offset)); }
  SYCL_EXTERNAL void set(size_t idx) { visited |= static_cast<uint64_t>(1) << (idx - offset); }
  SYCL_EXTERNAL void unset(size_t idx) { visited &= ~(static_cast<uint64_t>(1) << (idx - offset)); }
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
                                  const types::node_t* matching_order,
                                  const mbsm::DeviceBatchedQueryGraph& query_graphs,
                                  uint query_graph_id,
                                  const mbsm::DeviceBatchedDataGraph& data_graphs,
                                  uint data_graph_id) {
  for (int i = 0; i < depth; i++) {
    size_t query_nodes_offset = query_graphs.getPreviousNodes(query_graph_id);

    if (query_graphs.isNeighbor(matching_order[i] + query_nodes_offset, matching_order[depth] + query_nodes_offset)
        != data_graphs.isNeighbor(mapping[i], candidate)) {
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

utils::BatchedEvent joinCandidates(sycl::queue& queue,
                                   mbsm::DeviceBatchedQueryGraph& query_graphs,
                                   mbsm::DeviceBatchedDataGraph& data_graphs,
                                   mbsm::candidates::Candidates& candidates,
                                   mbsm::isomorphism::mapping::DGCR& dqcr,
                                   size_t* num_matches,
                                   bool find_first = true) {
  utils::BatchedEvent e;
  const size_t total_query_nodes = query_graphs.total_nodes;
  const size_t total_data_nodes = data_graphs.total_nodes;
  const size_t total_query_graphs = query_graphs.num_graphs;
  const size_t total_data_graphs = data_graphs.num_graphs;

  const size_t preferred_workgroup_size = 128; // TODO get from device
  const size_t subgroup_size = 32;             // TODO get from device

  sycl::nd_range<1> nd_range{total_data_graphs * preferred_workgroup_size, preferred_workgroup_size};

  auto e1 = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<device::kernels::JoinCandidatesKernel>(
        nd_range,
        [=, query_graphs = query_graphs, data_graphs = data_graphs, candidates = candidates.getCandidatesDevice(), dqcr = dqcr](
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

          uint32_t mapping[30];
          types::node_t matching_order[30];

          for (size_t data_graph_id = wgid; data_graph_id < total_data_graphs; data_graph_id += wg.get_group_linear_range()) {
            size_t start_data_graph = data_graphs.graph_offsets[data_graph_id];
            size_t end_data_graph = data_graphs.graph_offsets[data_graph_id + 1];

            size_t start_query = dqcr.data_graph_offsets[data_graph_id];
            size_t end_query = dqcr.data_graph_offsets[data_graph_id + 1];

            size_t num_data_nodes = end_data_graph - start_data_graph;
            auto data_graph_row_offsets = data_graphs.row_offsets + start_data_graph;
            auto data_graph_column_indices = data_graphs.column_indices;

            sycl::group_barrier(wg);


            for (size_t query_graph_it = wglid; query_graph_it < (end_query - start_query);
                 query_graph_it += wgsize) { // iterate over all query graphs
              size_t query_graph_id = dqcr.query_graph_indices[start_query + query_graph_it];
              const size_t offset_query_nodes = query_graphs.getPreviousNodes(query_graph_id);
              const uint num_query_nodes = query_graphs.getGraphNodes(query_graph_id);

              for (int i = 0; i < num_query_nodes; i++) { matching_order[i] = i; }

              Stack stack[30]; // TODO: assume max depth of 30 but make it dynamic

              // start DFS
              Visited visited{start_data_graph};
              uint top = 0;
              stack[top++] = {0, 0}; // initialize stack with the first node
              // DFS loop
              while (top > 0) {
                // get the top frame
                auto frame = stack[top - 1];
                auto query_node = matching_order[frame.depth];

                if (frame.depth == num_query_nodes) { // found a match and output solution
                  num_matches_ref++;
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
                  if (top > 0) { visited.unset(mapping[query_node]); }
                  continue;
                }

                // try the next candidate
                auto candidate = candidates.getCandidateAt(query_node + offset_query_nodes, frame.candidateIdx, start_data_graph, end_data_graph);
                // increment the candidate index for the next iteration
                stack[top - 1].candidateIdx++;


                if (!visited.get(candidate)
                    && (frame.depth == 0
                        || isValidMapping(candidate, frame.depth, mapping, matching_order, query_graphs, query_graph_id, data_graphs, wgid))) {
                  mapping[frame.depth] = candidate;
                  visited.set(candidate);
                  stack[top++] = {frame.depth + 1, 0};
                }
              }
            }
          }
        });
  });

  e.add(e1);
  return e;
}

} // namespace join
} // namespace isomorphism
} // namespace mbsm