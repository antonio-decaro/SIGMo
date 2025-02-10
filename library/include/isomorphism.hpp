#pragma once

#include "candidates.hpp"
#include "device.hpp"
#include "graph.hpp"
#include "pool.hpp"
#include "types.hpp"
#include <sycl/sycl.hpp>


namespace mbsm {
namespace isomorphism {
namespace filter {

// TODO consider to use the shared memory to store the graph to avoid uncoallesced memory access
sycl::event generateQuerySignatures(sycl::queue& queue, mbsm::DeviceBatchedQueryGraph& graphs, mbsm::candidates::Signature* signatures) {
  sycl::range<1> global_range(graphs.total_nodes);
  auto e = queue.submit([&](sycl::handler& cgh) {
    auto* adjacency = graphs.adjacency;
    auto* labels = graphs.labels;
    auto* graph_offsets = graphs.graph_offsets;
    auto* num_nodes = graphs.num_nodes;
    auto total_nodes = graphs.total_nodes;

    cgh.parallel_for<mbsm::device::kernels::GenerateQuerySignaturesKernel>(global_range, [=](sycl::item<1> item) {
      auto node_id = item.get_id(0);

      // Find the graph that the node belongs to
      uint32_t graph_id = mbsm::utils::binarySearch(num_nodes, graphs.num_graphs, node_id);

      // fetch the label, the adjacency matrix
      mbsm::types::label_t node_label = labels[node_id];
      mbsm::types::adjacency_t* adjacency_matrix = adjacency + graph_offsets[graph_id];

      // Calculate the number of nodes in the previous graphs and the number of nodes in the current graph
      size_t prev_nodes = graph_id ? num_nodes[graph_id - 1] : 0;
      auto graph_size = num_nodes[graph_id] - prev_nodes;

      // Calculate the number of adjacency integers required to store the adjacency matrix
      uint8_t adjacency_integers = mbsm::utils::getNumOfAdjacencyIntegers(graph_size);

      // Calculate the node index in the current graph
      types::node_t node = node_id - prev_nodes;

      // Get the neighbors of the current node
      types::node_t neighbors[4];
      mbsm::utils::adjacency_matrix::getNeighbors(adjacency_matrix, adjacency_integers, node, neighbors);

      // Initialize the signature for the current node
      signatures[node_id].setLabelCount(node_label, 1);
      for (uint8_t i = 0; neighbors[i] != types::NULL_NODE; ++i) {
        auto neighbor = neighbors[i] + prev_nodes;
        signatures[node_id].incrementLabelCount(labels[neighbor]);
      }
    });
  });
  return e;
}

sycl::event generateDataSignatures(sycl::queue& queue, mbsm::DeviceBatchedDataGraph& graphs, mbsm::candidates::Signature* signatures) {
  sycl::range<1> global_range(graphs.total_nodes);
  auto e = queue.submit([&](sycl::handler& cgh) {
    auto* row_offsets = graphs.row_offsets;
    auto* column_indices = graphs.column_indices;
    auto* labels = graphs.labels;

    cgh.parallel_for<mbsm::device::kernels::GenerateDataSignaturesKernel>(global_range, [=](sycl::item<1> item) {
      auto node_id = item.get_id(0);

      uint32_t start_neighbor = row_offsets[node_id];
      uint32_t end_neighbor = row_offsets[node_id + 1];
      mbsm::types::label_t node_label = labels[node_id];

      signatures[node_id].setLabelCount(node_label, 1);
      for (uint32_t i = start_neighbor; i < end_neighbor; ++i) {
        auto neighbor = column_indices[i];
        signatures[node_id].incrementLabelCount(labels[neighbor]);
      }
    });
  });
  return e;
}

// TODO maybe invert the parallelization domain (data nodes in the outer loop) to improve memory access and reduce the number of iterations (requires
// atomic operations on candidates)
sycl::event filterCandidates(sycl::queue& queue,
                             mbsm::DeviceBatchedQueryGraph& query_graph,
                             mbsm::DeviceBatchedDataGraph& data_graph,
                             mbsm::candidates::Signature* query_signatures,
                             mbsm::candidates::Signature* data_signatures,
                             mbsm::candidates::Candidates* candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;
  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<mbsm::device::kernels::FilterCandidatesKernel>(sycl::range<1>(total_query_nodes), [=](sycl::item<1> item) {
      auto query_node_id = item.get_id(0);
      auto query_signature = query_signatures[query_node_id];
      auto query_labels = query_graph.labels;

      if (query_labels[query_node_id] == static_cast<types::label_t>(1)) { // TODO optimize wildcard node filtering
        for (size_t data_node_id = 0; data_node_id < total_data_nodes; ++data_node_id) { candidates[query_node_id].insert(data_node_id); }
      } else {
        for (size_t data_node_id = 0; data_node_id < total_data_nodes; ++data_node_id) {
          auto data_signature = data_signatures[data_node_id];

          bool insert = true;
          for (types::label_t l = 0; l < 16; l++) {
            insert &= query_signature.getLabelCount(l) <= data_signature.getLabelCount(l);
            if (!insert) break;
          }
          if (insert) { candidates[query_node_id].insert(data_node_id); }
        }
      }
    });
  });

  return e;
}

} // namespace filter

} // namespace isomorphism
} // namespace mbsm