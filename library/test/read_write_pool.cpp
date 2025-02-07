#include <bitset>
#include <cassert>
#include <iostream>
#include <mbsm.hpp>

void printBinary(mbsm::types::adjacency_t num) {
  std::bitset<64> x(num);
  for (int i = 0; i < x.size(); ++i) { std::cout << x[i]; }
  std::cout << std::endl;
}

void compareGraphs(mbsm::DataGraph& g1, mbsm::DataGraph& g2) {
  assert(g1.getNumNodes() == g2.getNumNodes());
  for (size_t i = 0; i < g1.getNumNodes() + 1; ++i) { assert(g1.getRowOffsets()[i] == g2.getRowOffsets()[i]); }
  for (size_t i = 0; i < g1.getRowOffsets()[g1.getNumNodes()]; ++i) { assert(g1.getColumnIndices()[i] == g2.getColumnIndices()[i]); }
  for (size_t i = 0; i < g1.getNumNodes(); ++i) { assert(g1.getLabels()[i] == g2.getLabels()[i]); }
}

void compareGraphs(mbsm::QueryGraph& g1, mbsm::QueryGraph& g2) {
  assert(g1.getNumNodes() == g2.getNumNodes());
  int size = mbsm::utils::getNumOfAdjacencyIntegers(g1.getNumNodes());
  for (int i = 0; i < size; ++i) { assert(g1.getAdjacencyMatrix()[i] == g2.getAdjacencyMatrix()[i]); }
  for (int i = 0; i < g1.getNumNodes(); ++i) { assert(g1.getLabels()[i] == g2.getLabels()[i]); }
}

int main(int argc, char** argv) {
  std::string fname1 = std::string(argv[1]);
  std::string fname2 = std::string(argv[2]);
  std::vector<mbsm::QueryGraph> query_graphs = mbsm::io::loadQueryGraphsFromFile(fname1);
  std::vector<mbsm::DataGraph> data_graphs = mbsm::io::loadDataGraphsFromFile(fname2);

  mbsm::GraphPool write_pool(data_graphs, query_graphs);

  mbsm::io::savePoolToBinary(write_pool, "pool.bin");

  auto read_pool = mbsm::io::loadPoolFromBinary("pool.bin");

  assert(write_pool.getDataGraphs().size() == read_pool.getDataGraphs().size());
  assert(write_pool.getQueryGraphs().size() == read_pool.getQueryGraphs().size());

  for (size_t i = 0; i < write_pool.getDataGraphs().size(); ++i) { compareGraphs(write_pool.getDataGraphs()[i], read_pool.getDataGraphs()[i]); }

  for (size_t i = 0; i < write_pool.getQueryGraphs().size(); ++i) { compareGraphs(write_pool.getQueryGraphs()[i], read_pool.getQueryGraphs()[i]); }

  std::cout << "All tests passed!" << std::endl;

  return 0;
}