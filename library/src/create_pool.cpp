/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <mbsm.hpp>

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <data_graphs_file> <query_graphs_file> <output_file>" << std::endl;
    return 1;
  }

  std::string data_graphs_file = std::string(argv[1]);
  std::string query_graphs_file = std::string(argv[2]);
  std::string output_file = std::string(argv[3]);

  std::cout << "Reading query graphs from " << query_graphs_file << std::endl;
  std::vector<mbsm::QueryGraph> query_graphs = mbsm::io::loadQueryGraphsFromFile(query_graphs_file);
  std::cout << "Number of query graphs: " << query_graphs.size() << std::endl;

  std::cout << "Reading data graphs from " << data_graphs_file << std::endl;
  std::vector<mbsm::DataGraph> data_graphs = mbsm::io::loadDataGraphsFromFile(data_graphs_file);
  std::cout << "Number of data graphs: " << data_graphs.size() << std::endl;

  std::cout << "Creating pool" << std::endl;
  mbsm::GraphPool pool(data_graphs, query_graphs);

  std::cout << "Saving pool to " << output_file << std::endl;
  mbsm::io::savePoolToBinary(pool, output_file);

  return 0;
}