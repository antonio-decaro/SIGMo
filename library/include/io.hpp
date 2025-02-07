#pragma once

#include "graph.hpp"
#include "pool.hpp"
#include <fstream>
#include <iostream>
#include <string>

namespace mbsm {
namespace io {

std::vector<QueryGraph> loadQueryGraphsFromFile(const std::string& filename) {
  std::ifstream file(filename);
  std::string line;
  std::vector<mbsm::QueryGraph> query_graphs;
  while (std::getline(file, line)) {
    mbsm::IntermediateGraph intermediate_graph{line};
    query_graphs.push_back(intermediate_graph.toQueryGraph());
  }
  return query_graphs;
}

std::vector<DataGraph> loadDataGraphsFromFile(const std::string& filename) {
  std::ifstream file(filename);
  std::string line;
  std::vector<mbsm::DataGraph> data_graphs;
  while (std::getline(file, line)) {
    mbsm::IntermediateGraph intermediate_graph{line};
    data_graphs.push_back(intermediate_graph.toDataGraph());
  }
  return data_graphs;
}

mbsm::GraphPool loadPoolFromBinary(const std::string& filename) {
  mbsm::GraphPool pool;
  // TODO Implement logic to load GraphPool from binary file
  return pool;
}

void savePoolToBinary(const mbsm::GraphPool& pool, const std::string& filename) {
  // TODO Implement logic to save GraphPool to binary file
}

} // namespace io
} // namespace mbsm