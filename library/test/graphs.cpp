/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./include/utils.hpp"
#include "gtest/gtest.h"
#include <mbsm.hpp>

TEST(GraphTest, GetNeighbors) {
  std::string fname1 = std::string(TEST_QUERY_PATH);
  std::vector<mbsm::AMGraph> query_graphs = mbsm::io::loadQueryGraphsFromFile(fname1);

  mbsm::types::node_t neighbors[4];
  for (auto& graph : query_graphs) {
    for (int i = 0; i < graph.getNumNodes(); ++i) {
      mbsm::utils::adjacency_matrix::getNeighbors(
          graph.getAdjacencyMatrix(), mbsm::utils::getNumOfAdjacencyIntegers(graph.getNumNodes()), i, neighbors);
      // Add assertions to verify the correctness of neighbors
      ASSERT_TRUE(neighbors[0] != mbsm::types::NULL_NODE);
    }
  }
}

TEST(GraphTest, IntoQueryDevice) {
  std::string fname1 = std::string(TEST_QUERY_PATH);
  std::vector<mbsm::AMGraph> query_graphs = mbsm::io::loadQueryGraphsFromFile(fname1);

  size_t total_nodes = 0;
  for (auto& graph : query_graphs) { total_nodes += graph.getNumNodes(); }

  sycl::queue queue{sycl::gpu_selector_v};

  auto device_query_graph = mbsm::createDeviceAMGraph(queue, query_graphs);

  ASSERT_EQ(device_query_graph.total_nodes, total_nodes);
  ASSERT_EQ(device_query_graph.num_graphs, query_graphs.size());

  size_t adjacency_size = 0;
  for (auto& graph : query_graphs) {
    ASSERT_EQ(device_query_graph.graph_offsets[&graph - &query_graphs[0]], adjacency_size);

    size_t curr_size = mbsm::utils::getNumOfAdjacencyIntegers(graph.getNumNodes());
    for (size_t i = 0; i < curr_size; ++i) { ASSERT_EQ(device_query_graph.adjacency[adjacency_size + i], graph.getAdjacencyMatrix()[i]); }
    adjacency_size += curr_size;
  }

  // create a vector with all the labels
  std::vector<mbsm::types::label_t> labels(total_nodes);
  size_t offset = 0;
  for (auto& graph : query_graphs) {
    for (size_t i = 0; i < graph.getNumNodes(); i++) { labels[offset + i] = graph.getLabels()[i]; }
    offset += graph.getNumNodes();
  }

  for (size_t i = 0; i < device_query_graph.total_nodes; ++i) { ASSERT_EQ(device_query_graph.labels[i], labels[i]); }
}

TEST(GraphTest, IntoDataDevice) {
  std::string fname1 = std::string(TEST_DATA_PATH);
  std::vector<mbsm::CSRGraph> data_graphs = mbsm::io::loadDataGraphsFromFile(fname1);

  size_t total_nodes = 0;
  size_t total_edges = 0;
  for (auto& graph : data_graphs) {
    total_nodes += graph.getNumNodes();
    total_edges += graph.getRowOffsets()[graph.getNumNodes()];
  }

  sycl::queue queue{sycl::gpu_selector_v};

  auto device_data_graph = mbsm::createDeviceCSRGraph(queue, data_graphs);

  ASSERT_EQ(device_data_graph.total_nodes, total_nodes);
  ASSERT_EQ(device_data_graph.num_graphs, data_graphs.size());
  ASSERT_EQ(device_data_graph.row_offsets[0], 0);

  // create a vector with all the labels, row_offsets and column_indices
  std::vector<mbsm::types::row_offset_t> graph_offsets(data_graphs.size() + 1);
  std::vector<mbsm::types::row_offset_t> row_offsets(total_nodes + 1);
  std::vector<mbsm::types::col_index_t> column_indices(total_edges);
  std::vector<mbsm::types::label_t> labels(total_nodes);

  size_t ro_offset = 0;
  size_t col_offset = 0;
  size_t label_offset = 0;
  graph_offsets[0] = 0;

  for (int i = 0; i < data_graphs.size(); i++) {
    auto& graph = data_graphs[i];
    size_t num_nodes = graph.getNumNodes();
    size_t num_row_offsets = num_nodes + 1;
    size_t num_column_indices = graph.getRowOffsets()[num_nodes];

    graph_offsets[i + 1] = graph_offsets[i] + num_nodes;

    for (size_t j = 0; j < num_row_offsets; ++j) { row_offsets[ro_offset + j] = graph.getRowOffsets()[j] + col_offset; }

    for (size_t j = 0; j < num_column_indices; ++j) { column_indices[col_offset + j] = graph.getColumnIndices()[j] + label_offset; }

    for (size_t j = 0; j < num_nodes; ++j) { labels[label_offset + j] = graph.getLabels()[j]; }

    ro_offset += num_nodes;
    col_offset += num_column_indices;
    label_offset += num_nodes;
  }

  // Verify the unique graph
  ASSERT_EQ(row_offsets.size(), total_nodes + 1);
  ASSERT_EQ(column_indices.size(), total_edges);
  ASSERT_EQ(labels.size(), total_nodes);

  for (size_t i = 0; i < graph_offsets.size(); ++i) { ASSERT_EQ(graph_offsets[i], device_data_graph.graph_offsets[i]); }

  for (size_t i = 0; i < total_nodes; ++i) { ASSERT_EQ(labels[i], device_data_graph.labels[i]); }

  for (size_t i = 0; i < column_indices.size(); ++i) { ASSERT_EQ(column_indices[i], device_data_graph.column_indices[i]); }

  for (size_t i = 0; i < row_offsets.size(); ++i) { ASSERT_EQ(row_offsets[i], device_data_graph.row_offsets[i]); }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}