#pragma once

#include "types.hpp"
#include "utils.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace mbsm {

class QueryGraph {
public:
  QueryGraph(const QueryGraph& other) = default;

  QueryGraph(std::vector<types::adjacency_t> adjacency, std::vector<types::label_t> labels, uint8_t num_nodes)
      : _adjacency(adjacency), _labels(labels), _num_nodes(num_nodes) {}

  types::adjacency_t* getAdjacencyMatrix() { return _adjacency.data(); }
  types::label_t* getLabels() { return _labels.data(); }
  int getNumNodes() const { return _num_nodes; }

private:
  std::vector<types::adjacency_t> _adjacency;
  std::vector<types::label_t> _labels;
  uint8_t _num_nodes;
};

class DataGraph {
public:
  DataGraph(const DataGraph& other) = default;

  DataGraph(std::vector<types::row_offset_t> row_offsets,
            std::vector<types::col_index_t> column_indices,
            std::vector<types::label_t> labels,
            size_t num_nodes)
      : _row_offsets(row_offsets), _column_indices(column_indices), _labels(labels), _num_nodes(num_nodes) {}

  const types::row_offset_t* getRowOffsets() const { return _row_offsets.data(); }
  const types::col_index_t* getColumnIndices() const { return _column_indices.data(); }
  const types::label_t* getLabels() const { return _labels.data(); }
  size_t getNumNodes() const { return _num_nodes; }

private:
  std::vector<types::row_offset_t> _row_offsets;   // CSR row pointers
  std::vector<types::col_index_t> _column_indices; // CSR column data
  std::vector<types::label_t> _labels;
  size_t _num_nodes;
};

struct DeviceBatchedDataGraph {
  types::row_offset_t* graph_offsets;
  types::row_offset_t* row_offsets;
  types::col_index_t* column_indices;
  types::label_t* labels;
  uint32_t num_graphs;
  size_t total_nodes;
};

struct DeviceBatchedQueryGraph {
  types::adjacency_t* adjacency;
  types::label_t* labels;
  uint32_t* num_nodes;
  size_t total_nodes;
  uint32_t num_graphs;
  uint32_t* graph_offsets;
};


class IntermediateGraph {
public:
  IntermediateGraph() = default;
  IntermediateGraph(const std::string& graph_str) {
    std::istringstream iss(graph_str);
    std::string token;
    int curr_label;
    size_t num_nodes, num_labels, num_edges;

    // Read number of nodes
    iss >> token;
    token = token.substr(2);
    num_nodes = std::stoul(token);
    node_labels.resize(num_nodes);
    // Read labels
    iss >> token;
    token = token.substr(2);
    num_labels = std::stoul(token);

    for (size_t i = 0; i < num_nodes; ++i) {
      size_t node;
      types::label_t label;
      iss >> node >> curr_label;
      node_labels[node] = curr_label;
    }
    this->max_labels = num_labels;

    // Read number of edges
    iss >> token;
    token = token.substr(2);
    num_edges = std::stoul(token);
    edges.resize(num_edges);

    // Read edges
    for (size_t i = 0; i < num_edges; ++i) {
      types::node_t u, v;
      iss >> u >> v;
      edges[i] = std::make_pair(u, v);
    }
  }

  QueryGraph toQueryGraph() {
    size_t array_size = utils::getNumOfAdjacencyIntegers(this->node_labels.size());
    std::vector<types::adjacency_t> adjacency(array_size);
    for (auto& edge : edges) { utils::adjacency_matrix::setBit(adjacency.data(), array_size, edge.first, edge.second); }

    return QueryGraph(adjacency, node_labels, node_labels.size());
  }

  DataGraph toDataGraph() {
    std::vector<types::row_offset_t> row_offsets(node_labels.size() + 1);
    std::vector<types::col_index_t> column_indices;
    std::vector<types::label_t> labels(node_labels.size());

    row_offsets[0] = 0;
    for (size_t i = 0; i < edges.size(); ++i) {
      auto& edge = edges[i];
      row_offsets[edge.first + 1]++;
      row_offsets[edge.second + 1]++;
    }

    for (size_t i = 1; i < row_offsets.size(); ++i) { row_offsets[i] += row_offsets[i - 1]; }

    column_indices.resize(row_offsets.back());
    std::vector<size_t> current_pos(row_offsets.size(), 0);
    for (size_t i = 0; i < edges.size(); ++i) {
      auto& edge = edges[i];
      column_indices[row_offsets[edge.first] + current_pos[edge.first]] = edge.second;
      column_indices[row_offsets[edge.second] + current_pos[edge.second]] = edge.first;
      current_pos[edge.first]++;
      current_pos[edge.second]++;
    }

    return DataGraph(row_offsets, column_indices, node_labels, node_labels.size());
  }

  std::vector<std::pair<types::node_t, types::node_t>> edges;
  std::vector<types::label_t> node_labels;
  size_t max_labels;
};

// TODO offload on GPU
static DeviceBatchedDataGraph createDeviceDataGraph(sycl::queue& queue, std::vector<DataGraph>& data_graphs) {
  DeviceBatchedDataGraph device_data_graph;

  size_t total_nodes = 0;
  size_t total_edges = 0;

  for (auto& graph : data_graphs) {
    total_nodes += graph.getNumNodes();
    total_edges += graph.getRowOffsets()[graph.getNumNodes()];
  }

  device_data_graph.total_nodes = total_nodes;
  device_data_graph.graph_offsets = sycl::malloc_shared<types::row_offset_t>(data_graphs.size() + 1, queue);
  device_data_graph.row_offsets = sycl::malloc_shared<types::row_offset_t>(total_nodes + 1, queue);
  device_data_graph.column_indices = sycl::malloc_shared<types::col_index_t>(total_edges, queue);
  device_data_graph.labels = sycl::malloc_shared<types::label_t>(total_nodes, queue);
  device_data_graph.num_graphs = data_graphs.size();

  device_data_graph.graph_offsets[0] = 0;
  size_t ro_offset = 0;
  size_t col_offset = 0;
  size_t label_offset = 0;

  for (int graph_id = 0; graph_id < data_graphs.size(); graph_id++) {
    auto& graph = data_graphs[graph_id];
    size_t num_nodes = graph.getNumNodes();
    size_t num_row_offsets = num_nodes + 1;
    size_t num_column_indices = graph.getRowOffsets()[num_nodes];
    device_data_graph.graph_offsets[graph_id + 1] = device_data_graph.graph_offsets[graph_id] + num_nodes;

    for (size_t j = 0; j < num_row_offsets; ++j) { device_data_graph.row_offsets[ro_offset + j] = graph.getRowOffsets()[j] + col_offset; }

    for (size_t j = 0; j < num_column_indices; ++j) { device_data_graph.column_indices[col_offset + j] = graph.getColumnIndices()[j] + label_offset; }

    for (size_t j = 0; j < num_nodes; ++j) { device_data_graph.labels[label_offset + j] = graph.getLabels()[j]; }

    ro_offset += num_nodes;
    col_offset += num_column_indices;
    label_offset += num_nodes;
  }

  return device_data_graph;
}

static void destroyDeviceDataGraph(DeviceBatchedDataGraph& device_data_graph, sycl::queue& queue) {
  sycl::free(device_data_graph.row_offsets, queue);
  sycl::free(device_data_graph.column_indices, queue);
  sycl::free(device_data_graph.labels, queue);
  sycl::free(device_data_graph.graph_offsets, queue);
}

static size_t getDeviceDataGraphAllocSize(const DeviceBatchedDataGraph& device_data_graph) {
  return device_data_graph.total_nodes * sizeof(types::label_t) + (device_data_graph.num_graphs + 1) * sizeof(types::row_offset_t)
         + (device_data_graph.total_nodes + 1) * sizeof(types::row_offset_t)
         + device_data_graph.row_offsets[device_data_graph.total_nodes] * sizeof(types::col_index_t);
}

static size_t getDeviceDataGraphAllocSize(const std::vector<DataGraph>& data_graphs) {
  size_t total_nodes = 0;
  size_t total_edges = 0;

  for (auto& graph : data_graphs) {
    total_nodes += graph.getNumNodes();
    total_edges += graph.getRowOffsets()[graph.getNumNodes()];
  }

  return total_nodes * sizeof(types::label_t) + (data_graphs.size() + 1) * sizeof(types::row_offset_t)
         + (total_nodes + 1) * sizeof(types::row_offset_t) + total_edges * sizeof(types::col_index_t);
}


// TODO offload on GPU
static DeviceBatchedQueryGraph createDeviceQueryGraph(sycl::queue& queue, std::vector<QueryGraph>& query_graphs) {
  DeviceBatchedQueryGraph device_query_graph;

  device_query_graph.num_graphs = query_graphs.size();

  // fetching the adjacency matrix size
  sycl::buffer<uint8_t> adjacency_sizes(query_graphs.size());
  device_query_graph.num_nodes = sycl::malloc_shared<uint32_t>(query_graphs.size(), queue);
  sycl::host_accessor adjacency_sizes_hacc(adjacency_sizes);
  for (size_t i = 0; i < query_graphs.size(); ++i) {
    auto& graph = query_graphs[i];
    adjacency_sizes_hacc[i] = utils::getNumOfAdjacencyIntegers(graph.getNumNodes());
    device_query_graph.num_nodes[i] = i == 0 ? graph.getNumNodes() : device_query_graph.num_nodes[i - 1] + graph.getNumNodes();
  }

  // prefix sum to get graph offsets
  device_query_graph.graph_offsets = sycl::malloc_shared<uint32_t>(query_graphs.size(), queue);
  device_query_graph.graph_offsets[0] = 0;
  for (size_t i = 1; i < query_graphs.size(); ++i) {
    device_query_graph.graph_offsets[i] = device_query_graph.graph_offsets[i - 1] + adjacency_sizes_hacc[i - 1];
  }

  // allocate memory for adjacency matrix
  device_query_graph.adjacency = sycl::malloc_shared<types::adjacency_t>(
      device_query_graph.graph_offsets[query_graphs.size() - 1] + adjacency_sizes_hacc[query_graphs.size() - 1], queue);

  // allocate memory for labels
  size_t total_labels = device_query_graph.num_nodes[query_graphs.size() - 1];
  device_query_graph.total_nodes = total_labels;
  device_query_graph.labels = sycl::malloc_shared<types::label_t>(total_labels, queue);

  // copy data to device
  size_t nodes_offset = 0;
  for (auto& graph : query_graphs) {
    size_t graph_size = graph.getNumNodes();
    queue.copy(graph.getAdjacencyMatrix(),
               device_query_graph.adjacency + device_query_graph.graph_offsets[&graph - &query_graphs[0]],
               utils::getNumOfAdjacencyIntegers(graph_size));

    queue.copy(graph.getLabels(), device_query_graph.labels + nodes_offset, graph_size);
    nodes_offset += graph_size;
  }

  // wait for all copies to finish
  queue.wait_and_throw();

  return device_query_graph;
}

static void destroyDeviceQueryGraph(DeviceBatchedQueryGraph& device_query_graph, sycl::queue& queue) {
  sycl::free(device_query_graph.adjacency, queue);
  sycl::free(device_query_graph.labels, queue);
  sycl::free(device_query_graph.num_nodes, queue);
  sycl::free(device_query_graph.graph_offsets, queue);
}

static size_t getDeviceQueryGraphAllocSize(const DeviceBatchedQueryGraph& device_query_graph) {
  size_t adjacency_size = device_query_graph.graph_offsets[device_query_graph.num_graphs - 1] * sizeof(types::adjacency_t);
  size_t labels_size = device_query_graph.total_nodes * sizeof(types::label_t);
  size_t num_nodes_size = device_query_graph.num_graphs * sizeof(uint32_t);
  size_t graph_offsets_size = device_query_graph.num_graphs * sizeof(uint32_t);

  return adjacency_size + labels_size + num_nodes_size + graph_offsets_size;
}

static size_t getDeviceQueryGraphAllocSize(const std::vector<QueryGraph>& query_graphs) {
  size_t total_labels = 0;
  size_t total_adjacency = 0;

  for (auto& graph : query_graphs) {
    total_labels += graph.getNumNodes();
    total_adjacency += utils::getNumOfAdjacencyIntegers(graph.getNumNodes());
  }

  return total_labels * sizeof(types::label_t) + total_adjacency * sizeof(types::adjacency_t) + query_graphs.size() * sizeof(uint32_t)
         + query_graphs.size() * sizeof(uint32_t);
}

template<typename T>
static size_t getDeviceGraphAllocSize(const T& data) {
  if constexpr (std::is_same_v<T, DeviceBatchedQueryGraph>) {
    return getDeviceQueryGraphAllocSize(data);
  } else if constexpr (std::is_same_v<T, DeviceBatchedDataGraph>) {
    return getDeviceDataGraphAllocSize(data);
  } else {
    throw std::runtime_error("Unsupported type");
  }
}

template<typename T>
static size_t getDeviceGraphAllocSize(const std::vector<T>& data) {
  if constexpr (std::is_same_v<T, QueryGraph>) {
    return getDeviceQueryGraphAllocSize(data);
  } else if constexpr (std::is_same_v<T, DataGraph>) {
    return getDeviceDataGraphAllocSize(data);
  } else {
    throw std::runtime_error("Unsupported type");
  }
}

} // namespace mbsm
