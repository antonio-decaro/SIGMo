/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./include/utils.hpp"
#include "gtest/gtest.h"
#include <sigmo.hpp>

TEST(ReadWriteTest, LoadQueryGraphsFromFile) {
  std::string fname1 = std::string(TEST_QUERY_PATH);
  std::vector<sigmo::AMGraph> query_graphs = sigmo::io::loadAMGraphsFromFile(fname1);
  // Add assertions to verify the correctness of query_graphs
  ASSERT_TRUE(query_graphs.size() > 0);
}

TEST(ReadWriteTest, WriteReadComparePool) {
  std::string fname1 = std::string(TEST_QUERY_PATH);
  std::string fname2 = std::string(TEST_DATA_PATH);
  std::vector<sigmo::AMGraph> query_graphs = sigmo::io::loadAMGraphsFromFile(fname1);
  std::vector<sigmo::CSRGraph> data_graphs = sigmo::io::loadCSRGraphsFromFile(fname2);

  sigmo::GraphPool write_pool(data_graphs, query_graphs);

  sigmo::io::savePoolToBinary(write_pool, TEST_TMP_PATH);

  auto read_pool = sigmo::io::loadPoolFromBinary(TEST_TMP_PATH);

  ASSERT_EQ(write_pool.getDataGraphs().size(), read_pool.getDataGraphs().size());
  ASSERT_EQ(write_pool.getQueryGraphs().size(), read_pool.getQueryGraphs().size());

  for (size_t i = 0; i < write_pool.getDataGraphs().size(); ++i) { compareGraphs(write_pool.getDataGraphs()[i], read_pool.getDataGraphs()[i]); }

  for (size_t i = 0; i < write_pool.getQueryGraphs().size(); ++i) { compareGraphs(write_pool.getQueryGraphs()[i], read_pool.getQueryGraphs()[i]); }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}