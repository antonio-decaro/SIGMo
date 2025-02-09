#include "./include/utils.hpp"
#include "gtest/gtest.h"
#include <mbsm.hpp>

TEST(GraphTest, GetNeighbors) {
  std::string fname1 = std::string(TEST_QUERY_PATH);
  std::vector<mbsm::QueryGraph> query_graphs = mbsm::io::loadQueryGraphsFromFile(fname1);

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
  std::vector<mbsm::QueryGraph> query_graphs = mbsm::io::loadQueryGraphsFromFile(fname1);

  size_t total_nodes = 0;
  for (auto& graph : query_graphs) { total_nodes += graph.getNumNodes(); }

  sycl::queue queue{sycl::gpu_selector_v};

  auto device_query_graph = mbsm::createDeviceQueryGraph(queue, query_graphs);

  ASSERT_EQ(device_query_graph.total_nodes, total_nodes);

  size_t adjacency_size = 0;
  for (auto& graph : query_graphs) {
    ASSERT_EQ(device_query_graph.graph_offsets[&graph - &query_graphs[0]], adjacency_size);

    size_t curr_size = mbsm::utils::getNumOfAdjacencyIntegers(graph.getNumNodes());
    for (size_t i = 0; i < curr_size; ++i) { ASSERT_EQ(device_query_graph.adjacency[adjacency_size + i], graph.getAdjacencyMatrix()[i]); }
    adjacency_size += curr_size;
  }

  // create a vector with all the labels
  mbsm::types::label_t* labels = new mbsm::types::label_t[total_nodes];
  size_t offset = 0;
  for (auto& graph : query_graphs) {
    for (size_t i = 0; i < graph.getNumNodes(); i++) { labels[offset + i] = graph.getLabels()[i]; }
    offset += graph.getNumNodes();
  }

  for (size_t i = 0; i < device_query_graph.total_nodes; ++i) { ASSERT_EQ(device_query_graph.labels[i], labels[i]); }

  delete[] labels;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}