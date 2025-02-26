#pragma once
#include "candidates.hpp"
#include "device.hpp"
#include "types.hpp"
#include "utils.hpp"
#include <cstdint>
#include <sycl/sycl.hpp>

namespace mbsm {
namespace signature {
template<size_t Bits = 4>
struct Signature {
  uint64_t signature;

  Signature() : signature(0) {}

  Signature(uint64_t signature) : signature(signature) {}

  SYCL_EXTERNAL static uint16_t getMaxLabels() { return sizeof(signature) * 8 / Bits; }

  SYCL_EXTERNAL void setLabelCount(uint8_t label, uint8_t count) {
    if (label < (64 / Bits) && count < (1 << Bits)) {
      signature &= ~((static_cast<uint64_t>((1 << Bits) - 1)) << (label * Bits)); // Clear the bits for the label
      signature |= (static_cast<uint64_t>(count) << (label * Bits));              // Set the new count
    }
  }

  SYCL_EXTERNAL uint8_t getLabelCount(uint8_t label) const {
    if (label < (64 / Bits)) { return (signature >> (label * Bits)) & ((1 << Bits) - 1); }
    return 0;
  }

  SYCL_EXTERNAL void incrementLabelCount(uint8_t label, uint8_t add = 1) {
    if (label < (64 / Bits)) {
      uint8_t count = getLabelCount(label);
      if (count < ((1 << Bits) - 1)) { // Ensure count does not exceed max value
        setLabelCount(label, count + static_cast<uint8_t>(add));
      }
    }
  }
};

// TODO consider to use the shared memory to store the graph to avoid uncoallesced memory access
utils::BatchedEvent generateQuerySignatures(sycl::queue& queue, DeviceBatchedQueryGraph& graphs, Signature<>* signatures) {
  utils::BatchedEvent event;
  sycl::range<1> global_range(graphs.total_nodes);
  auto e = queue.submit([&](sycl::handler& cgh) {
    auto* adjacency = graphs.adjacency;
    auto* labels = graphs.labels;
    auto* graph_offsets = graphs.graph_offsets;
    auto* num_nodes = graphs.num_nodes;
    auto num_graphs = graphs.num_graphs;
    auto total_nodes = graphs.total_nodes;

    cgh.parallel_for<mbsm::device::kernels::GenerateQuerySignaturesKernel<1>>(sycl::range<1>{graphs.total_nodes}, [=](sycl::item<1> item) {
      auto node_id = item.get_id(0);
      // Find the graph that the node belongs to
      uint32_t graph_id = mbsm::utils::binarySearch(num_nodes, num_graphs, node_id);

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
      for (types::node_t i = 0; neighbors[i] != types::NULL_NODE && i < types::MAX_NEIGHBORS; ++i) {
        auto neighbor = neighbors[i] + prev_nodes;
        signatures[node_id].incrementLabelCount(labels[neighbor]);
      }
    });
  });
  event.add(e);

  return event;
}

utils::BatchedEvent
refineQuerySignatures(sycl::queue& queue, DeviceBatchedQueryGraph& graphs, Signature<>* signatures, Signature<>* tmp_buff, size_t iter = 1) {
  utils::BatchedEvent event;
  sycl::range<1> global_range(graphs.total_nodes);

  auto copy_event = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(graphs.total_nodes), [=](sycl::item<1> item) { tmp_buff[item] = signatures[item]; });
  });
  event.add(copy_event);
  copy_event.wait();

  auto refinement_event = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(copy_event);
    auto* adjacency = graphs.adjacency;
    auto* labels = graphs.labels;
    auto* graph_offsets = graphs.graph_offsets;
    auto* num_nodes = graphs.num_nodes;
    auto num_graphs = graphs.num_graphs;
    auto total_nodes = graphs.total_nodes;
    const uint16_t max_labels_count = mbsm::signature::Signature<>::getMaxLabels();

    cgh.parallel_for<mbsm::device::kernels::GenerateQuerySignaturesKernel<2>>(sycl::range<1>{graphs.total_nodes}, [=](sycl::item<1> item) {
      auto node_id = item.get_id(0);
      // Find the graph that the node belongs to
      uint32_t graph_id = mbsm::utils::binarySearch(num_nodes, num_graphs, node_id);

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
      for (types::node_t i = 0; neighbors[i] != types::NULL_NODE && i < types::MAX_NEIGHBORS; ++i) {
        auto neighbor = neighbors[i] + prev_nodes;
        for (types::label_t l = 0; l < max_labels_count; l++) {
          auto count = tmp_buff[neighbor].getLabelCount(l);
          if (l == node_label) { count -= iter; }
          if (count > 0) signatures[node_id].incrementLabelCount(l, count);
        }
      }
    });
  });
  event.add(refinement_event);

  return event;
};

utils::BatchedEvent generateDataSignatures(sycl::queue& queue, DeviceBatchedDataGraph& graphs, Signature<>* signatures) {
  utils::BatchedEvent event;
  sycl::range<1> global_range(graphs.total_nodes);
  sycl::buffer<mbsm::signature::Signature<>, 1> buffer(sycl::range{global_range});
  auto e = queue.submit([&](sycl::handler& cgh) {
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
  event.add(e);

  return event;
}

utils::BatchedEvent
refineDataSignatures(sycl::queue& queue, DeviceBatchedDataGraph& graphs, Signature<>* signatures, Signature<>* tmp_buff, size_t iter = 1) {
  utils::BatchedEvent event;
  sycl::range<1> global_range(graphs.total_nodes);

  auto copy_event = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(graphs.total_nodes), [=](sycl::item<1> item) { tmp_buff[item] = signatures[item]; });
  });
  event.add(copy_event);
  auto refine_event = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(copy_event);
    auto* row_offsets = graphs.row_offsets;
    auto* column_indices = graphs.column_indices;
    auto* labels = graphs.labels;

    cgh.parallel_for<mbsm::device::kernels::GenerateDataSignaturesKernel<2>>(global_range, [=](sycl::item<1> item) {
      auto node_id = item.get_id(0);

      uint32_t start_neighbor = row_offsets[node_id];
      uint32_t end_neighbor = row_offsets[node_id + 1];
      mbsm::types::label_t node_label = labels[node_id];

      for (uint32_t i = start_neighbor; i < end_neighbor; ++i) {
        auto neighbor = column_indices[i];
        for (types::label_t l = 0; l < Signature<>::getMaxLabels(); l++) {
          auto count = tmp_buff[neighbor].getLabelCount(l);
          if (l == node_label) { count -= iter; }
          if (count > 0) signatures[node_id].incrementLabelCount(l, count);
        }
      }
    });
  });
  event.add(refine_event);
  return event;
}

} // namespace signature
} // namespace mbsm