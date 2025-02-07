#include <bitset>
#include <iostream>
#include <mbsm.hpp>


int main(int argc, char** argv) {
  std::string fname1 = std::string(argv[1]);
  std::vector<mbsm::QueryGraph> query_graphs = mbsm::io::loadQueryGraphsFromFile(fname1);

  mbsm::types::node_t* neighbors = new mbsm::types::node_t[4];
  for (auto& graph : query_graphs) {
    std::cout << "Number of nodes: " << static_cast<int>(graph.getNumNodes()) << std::endl;
    for (int i = 0; i < graph.getNumNodes(); ++i) {
      mbsm::utils::adjacency_matrix::getNeighbors(
          graph.getAdjacencyMatrix(), mbsm::utils::getNumOfAdjacencyIntegers(graph.getNumNodes()), i, neighbors);
      int neighbor_count = 0;
      std::cout << "Neighbors of node " << i << ":" << std::endl;
      for (int j = 0; neighbors[j] != mbsm::types::NULL_NODE; ++j) { std::cout << static_cast<int>(neighbors[j]) << std::endl; }
    }
  }

  return 0;
}