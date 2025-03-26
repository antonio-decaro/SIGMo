/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gtest/gtest.h"
#include <bitset>
#include <iostream>
#include <sigmo.hpp>

void printBinary(sigmo::types::adjacency_t num) {
  std::bitset<64> x(num);
  for (int i = 0; i < x.size(); ++i) { std::cout << x[i]; }
  std::cout << std::endl;
}

void printDataGraph(const sigmo::CSRGraph& graph) {
  std::cout << "Row Offsets: ";
  for (size_t i = 0; i < graph.getNumNodes() + 1; ++i) { std::cout << graph.getRowOffsets()[i] << " "; }
  std::cout << std::endl;

  std::cout << "Column Indices: ";
  for (size_t i = 0; i < graph.getRowOffsets()[graph.getNumNodes()]; ++i) { std::cout << graph.getColumnIndices()[i] << " "; }
  std::cout << std::endl;

  std::cout << "node_labels: ";
  for (size_t i = 0; i < graph.getNumNodes(); ++i) { std::cout << static_cast<int>(graph.getNodeLabels()[i]) << " "; }
  std::cout << std::endl;
}

void compareGraphs(sigmo::CSRGraph& g1, sigmo::CSRGraph& g2) {
  ASSERT_EQ(g1.getNumNodes(), g2.getNumNodes());
  for (size_t i = 0; i < g1.getNumNodes() + 1; ++i) { ASSERT_EQ(g1.getRowOffsets()[i], g2.getRowOffsets()[i]); }
  for (size_t i = 0; i < g1.getRowOffsets()[g1.getNumNodes()]; ++i) { ASSERT_EQ(g1.getColumnIndices()[i], g2.getColumnIndices()[i]); }
  for (size_t i = 0; i < g1.getNumNodes(); ++i) { ASSERT_EQ(g1.getNodeLabels()[i], g2.getNodeLabels()[i]); }
}

void compareGraphs(sigmo::AMGraph& g1, sigmo::AMGraph& g2) {
  ASSERT_EQ(g1.getNumNodes(), g2.getNumNodes());
  int size = sigmo::utils::getNumOfAdjacencyIntegers(g1.getNumNodes());
  for (int i = 0; i < size; ++i) { ASSERT_EQ(g1.getAdjacencyMatrix()[i], g2.getAdjacencyMatrix()[i]); }
  for (int i = 0; i < g1.getNumNodes(); ++i) { ASSERT_EQ(g1.getNodeLabels()[i], g2.getNodeLabels()[i]); }
}