#pragma once

#include "types.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sycl/sycl.hpp>
#include <vector>

namespace mbsm {

class IntermediateGraph {
public:
  std::vector<std::pair<types::node_t, types::node_t>> _edges;
  std::vector<types::label_t> _node_labels;
};

class QueryGraph {
public:
private:
  std::vector<types::adjacency_t> _adjacency;
  std::vector<types::label_t> _labels;
  uint8_t _num_nodes;
};

class DataGraph {
public:
private:
  std::vector<types::row_offset_t> _row_offsets;   // CSR row pointers
  std::vector<types::col_index_t> _column_indices; // CSR column data
  std::vector<types::label_t> _labels;
  size_t _num_nodes;
};

class DeviceBatchedDataGraph {
public:
  types::row_offset_t* row_offsets;
  types::col_index_t* column_indices;
  types::label_t* labels;
  uint32_t num_graphs;
  uint32_t* graph_offsets;
};

class DeviceBatchedQueryGraph {
public:
  types::adjacency_t* adjacency;
  types::label_t* labels;
  uint32_t num_graphs;
  uint32_t* graph_offsets;
};

static DeviceBatchedDataGraph createDeviceDataGraph(sycl::queue& queue, std::vector<DataGraph>& data_graphs) {
  DeviceBatchedDataGraph device_data_graph;

  // TODO Implement logic to create DeviceBatchedDataGraph from DataGraphs
  // prefix sum to get graph offsets

  return device_data_graph;
}

static DeviceBatchedQueryGraph createDeviceQueryGraph(sycl::queue& queue, std::vector<QueryGraph>& query_graphs) {
  DeviceBatchedQueryGraph device_query_graph;

  // TODO Implement logic to create DeviceBatchedQueryGraph from DataGraphs
  // prefix sum to get graph offsets

  return device_query_graph;
}

} // namespace mbsm
