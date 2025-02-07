#include <bitset>
#include <iostream>
#include <mbsm.hpp>

void printBinary(mbsm::types::adjacency_t num) {
  std::bitset<64> x(num);
  for (int i = 0; i < x.size(); ++i) { std::cout << x[i]; }
  std::cout << std::endl;
}

void printDataGraph(const mbsm::DataGraph& graph) {
  std::cout << "Row Offsets: ";
  for (size_t i = 0; i < graph.getNumNodes() + 1; ++i) { std::cout << graph.getRowOffsets()[i] << " "; }
  std::cout << std::endl;

  std::cout << "Column Indices: ";
  for (size_t i = 0; i < graph.getRowOffsets()[graph.getNumNodes()]; ++i) { std::cout << graph.getColumnIndices()[i] << " "; }
  std::cout << std::endl;

  std::cout << "Labels: ";
  for (size_t i = 0; i < graph.getNumNodes(); ++i) { std::cout << static_cast<int>(graph.getLabels()[i]) << " "; }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  std::string fname1 = std::string(argv[1]);
  std::string fname2 = std::string(argv[2]);
  std::vector<mbsm::QueryGraph> query_graphs = mbsm::io::loadQueryGraphsFromFile(fname1);
  for (auto& graph : query_graphs) {
    int size = mbsm::utils::getNumOfAdjacencyIntegers(graph.getNumNodes());
    for (int i = 0; i < size; ++i) {
      std::cout << "Adjacency matrix " << i << ": ";
      printBinary(graph.getAdjacencyMatrix()[i]);
    }
  }

  std::vector<mbsm::DataGraph> graphs = mbsm::io::loadDataGraphsFromFile(fname2);
  for (const auto& graph : graphs) { printDataGraph(graph); }

  return 0;
}