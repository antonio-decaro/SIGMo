#include "./include/utils.hpp"
#include "gtest/gtest.h"
#include <mbsm.hpp>

TEST(NeighborsTest, GetNeighbors) {
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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}