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

namespace detail {

template<typename KernelName, typename LambdaT>
struct IterateOverQueryNodes {
  void operator()(sycl::item<1> item) const {
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
    types::node_t neighbors[types::MAX_NEIGHBORS];
    mbsm::utils::adjacency_matrix::getNeighbors(adjacency_matrix, adjacency_integers, node, neighbors);

    lambda(node_id, graph_id, node_label, prev_nodes, graph_size, adjacency_integers, node, neighbors);
  }
  types::adjacency_t* adjacency;
  types::label_t* labels;
  types::node_t* graph_offsets;
  types::node_t* num_nodes;
  size_t total_nodes;
  const LambdaT&& lambda;
};

} // namespace detail

// TODO consider to use the shared memory to store the graph to avoid uncoallesced memory access
utils::BatchedEvent generateQuerySignatures(sycl::queue& queue,
                                            mbsm::DeviceBatchedQueryGraph& graphs,
                                            mbsm::candidates::Signature<>* signatures,
                                            size_t refinement_steps = 1) {
  utils::BatchedEvent event;
  sycl::event e;
  sycl::range<1> global_range(graphs.total_nodes);
  sycl::buffer<mbsm::candidates::Signature<>, 1> buffer(sycl::range{global_range});
  e = queue.submit([&](sycl::handler& cgh) {
    auto* adjacency = graphs.adjacency;
    auto* labels = graphs.labels;
    auto* graph_offsets = graphs.graph_offsets;
    auto* num_nodes = graphs.num_nodes;
    auto total_nodes = graphs.total_nodes;

    cgh.parallel_for<mbsm::device::kernels::GenerateQuerySignaturesKernel<1>>(global_range, [=](sycl::item<1> item) {
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
      types::node_t neighbors[types::MAX_NEIGHBORS];
      mbsm::utils::adjacency_matrix::getNeighbors(adjacency_matrix, adjacency_integers, node, neighbors);

      // Initialize the signature for the current node
      for (uint8_t i = 0; neighbors[i] != types::NULL_NODE && i < types::MAX_NEIGHBORS; ++i) {
        auto neighbor = neighbors[i] + prev_nodes;
        signatures[node_id].incrementLabelCount(labels[neighbor]);
      }
    });
  });
  event.add(e);
  for (int step = 0; step < refinement_steps; step++) {
    e = queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(event.getLastEvent());
      sycl::accessor acc(buffer, cgh, sycl::write_only);

      cgh.parallel_for(sycl::range<1>(graphs.total_nodes), [=](sycl::item<1> item) { acc[item] = signatures[item]; });
    });
    event.add(e);
    e = queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(event.getLastEvent());
      sycl::accessor acc(buffer, cgh, sycl::read_only);
      auto* adjacency = graphs.adjacency;
      auto* labels = graphs.labels;
      auto* graph_offsets = graphs.graph_offsets;
      auto* num_nodes = graphs.num_nodes;
      auto total_nodes = graphs.total_nodes;
      const uint16_t max_labels_count = mbsm::candidates::Signature<>::getMaxLabels();

      cgh.parallel_for<mbsm::device::kernels::GenerateQuerySignaturesKernel<2>>(global_range, [=](sycl::item<1> item) {
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
        types::node_t neighbors[types::MAX_NEIGHBORS];
        mbsm::utils::adjacency_matrix::getNeighbors(adjacency_matrix, adjacency_integers, node, neighbors);

        // Initialize the signature for the current node
        for (uint8_t i = 0; neighbors[i] != types::NULL_NODE && i < types::MAX_NEIGHBORS; ++i) {
          auto neighbor = neighbors[i] + prev_nodes;
          for (types::node_t l = 0; l < max_labels_count; l++) {
            auto count = acc[neighbor].getLabelCount(l);
            // if (l == node_label) { count -= 1; }
            signatures[node_id].incrementLabelCount(l, count);
          }
        }
      });
    });
    event.add(e);
  }
  return event;
}

utils::BatchedEvent generateDataSignatures(sycl::queue& queue,
                                           mbsm::DeviceBatchedDataGraph& graphs,
                                           mbsm::candidates::Signature<>* signatures,
                                           size_t refinement_steps = 1) {
  utils::BatchedEvent event;
  sycl::event e;
  sycl::range<1> global_range(graphs.total_nodes);
  sycl::buffer<mbsm::candidates::Signature<>, 1> buffer(sycl::range{global_range});
  e = queue.submit([&](sycl::handler& cgh) {
    auto* row_offsets = graphs.row_offsets;
    auto* column_indices = graphs.column_indices;
    auto* labels = graphs.labels;

    cgh.parallel_for<mbsm::device::kernels::GenerateDataSignaturesKernel<1>>(global_range, [=](sycl::item<1> item) {
      auto node_id = item.get_id(0);

      uint32_t start_neighbor = row_offsets[node_id];
      uint32_t end_neighbor = row_offsets[node_id + 1];
      mbsm::types::label_t node_label = labels[node_id];

      for (uint32_t i = start_neighbor; i < end_neighbor; ++i) {
        auto neighbor = column_indices[i];
        signatures[node_id].incrementLabelCount(labels[neighbor]);
      }
    });
  });
  for (int step = 0; step < refinement_steps; step++) {
    event.add(e);
    e = queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(event.getLastEvent());
      sycl::accessor acc(buffer, cgh, sycl::write_only);

      cgh.parallel_for(sycl::range<1>(graphs.total_nodes), [=](sycl::item<1> item) { acc[item] = signatures[item]; });
    });
    event.add(e);
    e = queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(event.getLastEvent());
      auto* row_offsets = graphs.row_offsets;
      auto* column_indices = graphs.column_indices;
      auto* labels = graphs.labels;
      sycl::accessor acc(buffer, cgh, sycl::read_only);

      cgh.parallel_for<mbsm::device::kernels::GenerateDataSignaturesKernel<2>>(global_range, [=](sycl::item<1> item) {
        auto node_id = item.get_id(0);

        uint32_t start_neighbor = row_offsets[node_id];
        uint32_t end_neighbor = row_offsets[node_id + 1];
        mbsm::types::label_t node_label = labels[node_id];

        for (uint32_t i = start_neighbor; i < end_neighbor; ++i) {
          auto neighbor = column_indices[i];
          for (types::label_t l = 0; l < candidates::Signature<>::getMaxLabels(); l++) {
            auto count = acc[neighbor].getLabelCount(l);
            if (l == node_label) { count -= 1; }
            if (count > 0) signatures[node_id].incrementLabelCount(l, count);
          }
        }
      });
    });
    event.add(e);
  }
  return event;
}

// TODO maybe invert the parallelization domain (data nodes in the outer loop) to improve memory access and reduce the number of iterations (requires
// atomic operations on candidates)
utils::BatchedEvent filterCandidates(sycl::queue& queue,
                                     mbsm::DeviceBatchedQueryGraph& query_graph,
                                     mbsm::DeviceBatchedDataGraph& data_graph,
                                     mbsm::candidates::Signature<>* query_signatures,
                                     mbsm::candidates::Signature<>* data_signatures,
                                     mbsm::candidates::Candidates& candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;
  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<mbsm::device::kernels::FilterCandidatesKernel>(
        sycl::range<1>(total_query_nodes), [=, candidates = candidates](sycl::item<1> item) {
          auto query_node_id = item.get_id(0);
          auto query_signature = query_signatures[query_node_id];
          auto query_labels = query_graph.labels;
          auto data_labels = data_graph.labels;

          // if (query_labels[query_node_id] == static_cast<types::label_t>(1)) { // TODO optimize wildcard node filtering
          //   for (size_t data_node_id = 0; data_node_id < total_data_nodes; ++data_node_id) { candidates.insert(query_node_id, data_node_id); }
          // } else {
          for (size_t data_node_id = 0; data_node_id < total_data_nodes; ++data_node_id) {
            if (query_labels[query_node_id] != data_labels[data_node_id]) { continue; }
            auto data_signature = data_signatures[data_node_id];

            bool insert = true;
            for (types::label_t l = 0; l < 16; l++) {
              insert &= query_signature.getLabelCount(l) <= data_signature.getLabelCount(l);
              if (!insert) break;
            }
            if (insert) { candidates.insert(query_node_id, data_node_id); }
          }
          // }
        });
  });
  utils::BatchedEvent be;
  be.add(e);
  return be;
}

} // namespace filter

} // namespace isomorphism
} // namespace mbsm