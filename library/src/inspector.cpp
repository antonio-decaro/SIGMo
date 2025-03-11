/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <mbsm.hpp>

template<typename T>
size_t countNodes(const std::vector<T>& graphs) {
  size_t count = 0;
  for (auto& graph : graphs) { count += graph.getNumNodes(); }
  return count;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << "<query|data|pool> <file>" << std::endl;
    return 1;
  }

  bool verbose = false;

  std::string mode = std::string(argv[1]);
  std::string file = std::string(argv[2]);
  if (argc > 3) { verbose = std::string(argv[3]) == "-v"; }

  std::cout << "Reading from " << file << std::endl;

  if (mode == "query") {
    std::vector<mbsm::AMGraph> query_graphs = mbsm::io::loadQueryGraphsFromFile(file);
    std::cout << "Number of query graphs: " << query_graphs.size() << std::endl;
    std::cout << "Number of nodes: " << countNodes(query_graphs) << std::endl;
    if (verbose)
      for (auto& query_graph : query_graphs) { std::cout << "Query graph with " << query_graph.getNumNodes() << " nodes" << std::endl; }
  } else if (mode == "data") {
    std::vector<mbsm::CSRGraph> data_graphs = mbsm::io::loadDataGraphsFromFile(file);
    std::cout << "Number of data graphs: " << data_graphs.size() << std::endl;
    std::cout << "Number of nodes: " << countNodes(data_graphs) << std::endl;
    if (verbose)
      for (auto& data_graph : data_graphs) { std::cout << "Data graph with " << data_graph.getNumNodes() << " nodes" << std::endl; }
  } else if (mode == "pool") {
    mbsm::GraphPool pool = mbsm::io::loadPoolFromBinary(file);
    std::cout << "Number of data graphs: " << pool.getDataGraphs().size() << std::endl;
    std::cout << "Number of query nodes: " << countNodes(pool.getQueryGraphs()) << std::endl;
    std::cout << "Number of query graphs: " << pool.getQueryGraphs().size() << std::endl;
    std::cout << "Number of data nodes: " << countNodes(pool.getDataGraphs()) << std::endl;
  } else {
    std::cerr << "Invalid mode: " << mode << std::endl;
    return 1;
  }
}