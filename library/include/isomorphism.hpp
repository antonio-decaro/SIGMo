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
utils::BatchedEvent generateQuerySignatures(sycl::queue& queue, mbsm::DeviceBatchedQueryGraph& graphs, mbsm::candidates::Signature<>* signatures) {
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
      for (uint8_t i = 0; neighbors[i] != types::NULL_NODE && i < types::MAX_NEIGHBORS; ++i) {
        auto neighbor = neighbors[i] + prev_nodes;
        signatures[node_id].incrementLabelCount(labels[neighbor]);
      }
    });
  });
  event.add(e);

  return event;
}

utils::BatchedEvent refineQuerySignatures(sycl::queue& queue,
                                          mbsm::DeviceBatchedQueryGraph& graphs,
                                          mbsm::candidates::Signature<>* signatures,
                                          mbsm::candidates::Signature<>* tmp_buff) {
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
    const uint16_t max_labels_count = mbsm::candidates::Signature<>::getMaxLabels();

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
      for (uint8_t i = 0; neighbors[i] != types::NULL_NODE && i < types::MAX_NEIGHBORS; ++i) {
        auto neighbor = neighbors[i] + prev_nodes;
        for (types::node_t l = 0; l < max_labels_count; l++) {
          auto count = tmp_buff[neighbor].getLabelCount(l);
          if (l == node_label) { count -= 1; }
          if (count > 0) signatures[node_id].incrementLabelCount(l, count);
        }
      }
    });
  });
  event.add(refinement_event);

  return event;
};

utils::BatchedEvent generateDataSignatures(sycl::queue& queue, mbsm::DeviceBatchedDataGraph& graphs, mbsm::candidates::Signature<>* signatures) {
  utils::BatchedEvent event;
  sycl::range<1> global_range(graphs.total_nodes);
  sycl::buffer<mbsm::candidates::Signature<>, 1> buffer(sycl::range{global_range});
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

utils::BatchedEvent refineDataSignatures(sycl::queue& queue,
                                         mbsm::DeviceBatchedDataGraph& graphs,
                                         mbsm::candidates::Signature<>* signatures,
                                         mbsm::candidates::Signature<>* tmp_buff) {
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
        for (types::label_t l = 0; l < candidates::Signature<>::getMaxLabels(); l++) {
          auto count = tmp_buff[neighbor].getLabelCount(l);
          if (l == node_label) { count -= 1; }
          if (count > 0) signatures[node_id].incrementLabelCount(l, count);
        }
      }
    });
  });
  event.add(refine_event);
  return event;
}

utils::BatchedEvent filterCandidates(sycl::queue& queue,
                                     mbsm::DeviceBatchedQueryGraph& query_graph,
                                     mbsm::DeviceBatchedDataGraph& data_graph,
                                     mbsm::candidates::Signature<>* query_signatures,
                                     mbsm::candidates::Signature<>* data_signatures,
                                     mbsm::candidates::Candidates candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;
  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<mbsm::device::kernels::FilterCandidatesKernel>(sycl::range<1>(total_data_nodes), [=](sycl::item<1> item) {
      auto data_node_id = item.get_id(0);
      auto data_signature = data_signatures[data_node_id];
      auto query_labels = query_graph.labels;
      auto data_labels = data_graph.labels;

      for (size_t query_node_id = 0; query_node_id < total_query_nodes; ++query_node_id) {
        if (query_labels[query_node_id] != data_labels[data_node_id]) { continue; }
        auto query_signature = query_signatures[query_node_id];

        bool insert = true;
        for (types::label_t l = 0; l < candidates::Signature<>::getMaxLabels(); l++) {
          insert &= query_signature.getLabelCount(l) <= data_signature.getLabelCount(l);
          if (!insert) break;
        }
        if (insert) { candidates.atomicInsert(query_node_id, data_node_id); }
      }
    });
  });

  utils::BatchedEvent be;
  be.add(e);
  return be;
}

utils::BatchedEvent refineCandidates(sycl::queue& queue,
                                     mbsm::DeviceBatchedQueryGraph& query_graph,
                                     mbsm::DeviceBatchedDataGraph& data_graph,
                                     mbsm::candidates::Signature<>* query_signatures,
                                     mbsm::candidates::Signature<>* data_signatures,
                                     mbsm::candidates::Candidates candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;
  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<mbsm::device::kernels::RefineCandidatesKernel>(sycl::range<1>(total_data_nodes), [=](sycl::item<1> item) {
      auto data_node_id = item.get_id(0);
      auto data_signature = data_signatures[data_node_id];
      auto query_labels = query_graph.labels;
      auto data_labels = data_graph.labels;

      for (size_t query_node_id = 0; query_node_id < total_query_nodes; ++query_node_id) {
        if (!candidates.atomicContains(query_node_id, data_node_id)) { continue; }
        auto query_signature = query_signatures[query_node_id];

        bool keep = data_labels[data_node_id] == query_labels[query_node_id];
        for (types::label_t l = 0; l < candidates::Signature<>::getMaxLabels() && keep; l++) {
          keep = keep && (query_signature.getLabelCount(l) <= data_signature.getLabelCount(l));
        }
        if (!keep) { candidates.atomicRemove(query_node_id, data_node_id); }
      }
    });
  });

  utils::BatchedEvent be;
  be.add(e);
  return be;
}

} // namespace filter

} // namespace isomorphism
} // namespace mbsm