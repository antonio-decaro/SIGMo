#include "gtest/gtest.h"
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

void compareGraphs(mbsm::DataGraph& g1, mbsm::DataGraph& g2) {
  ASSERT_EQ(g1.getNumNodes(), g2.getNumNodes());
  for (size_t i = 0; i < g1.getNumNodes() + 1; ++i) { ASSERT_EQ(g1.getRowOffsets()[i], g2.getRowOffsets()[i]); }
  for (size_t i = 0; i < g1.getRowOffsets()[g1.getNumNodes()]; ++i) { ASSERT_EQ(g1.getColumnIndices()[i], g2.getColumnIndices()[i]); }
  for (size_t i = 0; i < g1.getNumNodes(); ++i) { ASSERT_EQ(g1.getLabels()[i], g2.getLabels()[i]); }
}

void compareGraphs(mbsm::QueryGraph& g1, mbsm::QueryGraph& g2) {
  ASSERT_EQ(g1.getNumNodes(), g2.getNumNodes());
  int size = mbsm::utils::getNumOfAdjacencyIntegers(g1.getNumNodes());
  for (int i = 0; i < size; ++i) { ASSERT_EQ(g1.getAdjacencyMatrix()[i], g2.getAdjacencyMatrix()[i]); }
  for (int i = 0; i < g1.getNumNodes(); ++i) { ASSERT_EQ(g1.getLabels()[i], g2.getLabels()[i]); }
}