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
                                     mbsm::signature::Signature<>* query_signatures,
                                     mbsm::signature::Signature<>* data_signatures,
                                     mbsm::candidates::Candidates candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;
  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<mbsm::device::kernels::FilterCandidatesKernel<D>>(sycl::range<1>(total_data_nodes), [=](sycl::item<1> item) {
      auto data_node_id = item.get_id(0);
      auto data_signature = data_signatures[data_node_id];
      auto query_labels = query_graph.labels;
      auto data_labels = data_graph.labels;

      for (size_t query_node_id = 0; query_node_id < total_query_nodes; ++query_node_id) {
        if (query_labels[query_node_id] != data_labels[data_node_id]) { continue; }
        auto query_signature = query_signatures[query_node_id];

        bool insert = true;
        for (types::label_t l = 0; l < signature::Signature<>::getMaxLabels(); l++) {
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
                                     mbsm::signature::Signature<>* query_signatures,
                                     mbsm::signature::Signature<>* data_signatures,
                                     mbsm::candidates::Candidates candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;
  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<mbsm::device::kernels::RefineCandidatesKernel<D>>(sycl::range<1>(total_data_nodes), [=](sycl::item<1> item) {
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
        for (types::label_t l = 0; l < signature::Signature<>::getMaxLabels() && keep; l++) {
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

namespace join {
/**
 * // TODO improve data locality and shared memory usage
 * // TODO design data structure for temporary store the candidates for each query graph in shared memory
 * // TODO improve readibility
 * // TODO fix bugs
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
                                  uint32_t* mapping,
                                  const mbsm::DeviceBatchedQueryGraph& query_graphs,
                                  uint query_graph_id,
                                  const mbsm::DeviceBatchedDataGraph& data_graphs,
                                  uint data_graph_id) {
  for (int i = 0; i < depth; i++) {
    auto query_node = i;
    auto data_node = mapping[i];

    size_t query_nodes_offset = query_graphs.getPreviousNodes(query_graph_id);

    if (query_graphs.isNeighbor(query_node + query_nodes_offset, depth + query_nodes_offset) && !data_graphs.isNeighbor(data_node, candidate)) {
      return false;
    }
  }

  return true;
}

utils::BatchedEvent joinCandidates(sycl::queue& queue,
                                   mbsm::DeviceBatchedQueryGraph& query_graphs,
                                   mbsm::DeviceBatchedDataGraph& data_graphs,
                                   mbsm::candidates::Candidates candidates) {
  utils::BatchedEvent e;
  const size_t total_query_nodes = query_graphs.total_nodes;
  const size_t total_data_nodes = data_graphs.total_nodes;
  const size_t total_query_graphs = query_graphs.num_graphs;
  const size_t total_data_graphs = data_graphs.num_graphs;

  const size_t preferred_workgroup_size = 32; // TODO get from device
  const size_t subgroup_size = 32;            // TODO get from device

  sycl::nd_range<1> nd_range{total_data_graphs * preferred_workgroup_size, preferred_workgroup_size};

  sycl::buffer<Mapping, 1> solution_buf{sycl::range<1>{10}}; // TODO make it dynamic
  sycl::buffer<uint, 1> solution_tail_buf{sycl::range<1>{1}};

  auto e1 = queue.submit([&](sycl::handler& cgh) {
    sycl::accessor solution_acc{solution_buf, cgh, sycl::read_write};
    sycl::accessor solution_tail_acc{solution_tail_buf, cgh, sycl::read_write};

    cgh.parallel_for(nd_range, [=, query_graphs = query_graphs, data_graphs = data_graphs](sycl::nd_item<1> item) {
      sycl::atomic_ref<uint, sycl::memory_order::relaxed, sycl::memory_scope::device> solution_tail{solution_tail_acc[0]};

      const size_t lid = item.get_local_linear_id();
      const size_t gid = item.get_global_linear_id();

      const auto wg = item.get_group();
      const size_t wgid = wg.get_group_linear_id();
      const size_t wglid = wg.get_local_linear_id();

      const auto sg = item.get_sub_group();
      const size_t sgid = sg.get_group_linear_id();
      const size_t sglid = sg.get_local_linear_id();
      const size_t sgsize = sg.get_local_range()[0];

      size_t start_data_graph = data_graphs.graph_offsets[wgid];
      size_t end_data_graph = data_graphs.graph_offsets[wgid + 1];
      size_t num_data_nodes = end_data_graph - start_data_graph;
      auto data_graph_row_offsets = data_graphs.row_offsets + start_data_graph;
      auto data_graph_column_indices = data_graphs.column_indices;

      if (gid == 0) { solution_tail = 0; }
      sycl::group_barrier(wg);

      Stack stack[12]; // TODO: assume max depth of 12 but make it dynamic
      uint32_t mapping[12];

      for (size_t query_graph_id = sgid; query_graph_id < total_query_graphs;
           query_graph_id += sg.get_group_linear_range()) { // iterate over all query graphs
        const size_t offset_query_nodes = query_graphs.getPreviousNodes(query_graph_id);
        const uint num_query_nodes = query_graphs.getGraphNodes(query_graph_id);
        // int starting_node = sglid;
        // size_t starting_node_candidates = sglid < num_query_nodes ? candidates.getCandidatesCount(sglid, start_data_graph, end_data_graph) : 0;

        // // elect the starting node according to the number of candidates
        // // TODO elect also the other nodes
        // starting_node_candidates
        //     = sycl::reduce_over_group(sg, starting_node_candidates, sycl::maximum<>()); // the starting node is the one with the most candidates
        // if (sglid >= num_query_nodes || starting_node_candidates != candidates.getCandidatesCount(sglid, start_data_graph, end_data_graph)) {
        //   starting_node = -1;
        // }
        // starting_node = sycl::reduce_over_group(sg, starting_node, sycl::maximum<>());
        // if (starting_node == -1) { continue; } // no starting node
        int starting_node = 0;
        size_t starting_node_candidates = candidates.getCandidatesCount(starting_node + offset_query_nodes, start_data_graph, end_data_graph);

        for (size_t target_root_id = sglid; target_root_id < starting_node_candidates; target_root_id += sgsize) {
          // start DFS
          Visited visited{start_data_graph};
          uint top = 0;
          auto target_root = candidates.getCandidateAt(starting_node + offset_query_nodes, target_root_id, start_data_graph, end_data_graph);
          mapping[0] = target_root;
          visited.set(target_root);
          stack[top++] = {1, 0}; // initialize stack with the first node
          // DFS loop
          while (top > 0) {
            // get the top frame
            auto frame = stack[top - 1];

            if (frame.depth == num_query_nodes) { // found a match and output solution
              {
                auto solution_idx = solution_tail++;
                solution_acc[solution_idx].query_graph_id = query_graph_id;
                solution_acc[solution_idx].data_graph_id = wgid;
                for (int i = 0; i < num_query_nodes; i++) {
                  solution_acc[solution_idx].query_nodes[i] = i;
                  solution_acc[solution_idx].data_nodes[i] = mapping[i];
                }
                solution_acc[solution_idx].query_nodes[num_query_nodes] = -1;
                solution_acc[solution_idx].data_nodes[num_query_nodes] = -1;
              }
              top--;
              continue;
            }
            // no more candidates
            if (frame.candidateIdx >= candidates.getCandidatesCount(frame.depth + offset_query_nodes, start_data_graph, end_data_graph)) {
              // backtrack
              top--;
              // free the failed mapping
              if (top > 0) { visited.unset(mapping[frame.depth]); }
              continue;
            }

            // try the next candidate
            auto candidate = candidates.getCandidateAt(frame.depth + offset_query_nodes, frame.candidateIdx, start_data_graph, end_data_graph);
            // increment the candidate index for the next iteration
            stack[top - 1].candidateIdx++;

            if (!visited.get(candidate) && isValidMapping(candidate, frame.depth, mapping, query_graphs, query_graph_id, data_graphs, wgid)) {
              mapping[frame.depth] = candidate;
              visited.set(candidate);
              stack[top++] = {frame.depth + 1, 0};
            }
          }
        }
      }
    });
  });

  e1.wait();
  sycl::host_accessor acc{solution_buf};
  sycl::host_accessor tail{solution_tail_buf};
  std::cout << "Solutions: " << std::endl;
  for (int i = 0; i < tail[0]; i++) {
    std::cout << "\t" << acc[i].query_graph_id << " -> " << acc[i].data_graph_id << ": ";
    for (int j = 0; j < 12; j++) {
      if (acc[i].query_nodes[j] == -1) { break; }
      std::cout << acc[i].query_nodes[j] << " -> " << acc[i].data_nodes[j] << ", ";
    }
    std::cout << std::endl;
  }

  e.add(e1);
  return e;
}

} // namespace join

} // namespace isomorphism
} // namespace mbsm