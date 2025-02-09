#include <iostream>
#include <mbsm.hpp>


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
  std::string pool_file = std::string(argv[1]);

  std::cout << "Reading pool from " << pool_file << std::endl;
  mbsm::GraphPool pool = mbsm::io::loadPoolFromBinary(argv[1]);

  std::cout << "Number of data graphs: " << pool.getDataGraphs().size() << std::endl;
  std::cout << "Number of query graphs: " << pool.getQueryGraphs().size() << std::endl;

  auto data_graph = pool.getDataGraphs()[4125];
  printDataGraph(data_graph);
}