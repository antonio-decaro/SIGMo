/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "graph.hpp"
#include "pool.hpp"
#include <fstream>
#include <iostream>
#include <string>

namespace sigmo {
namespace io {

std::vector<AMGraph> loadAMGraphsFromFile(const std::string& filename) {
  std::ifstream file(filename);
  std::string line;
  std::vector<sigmo::AMGraph> query_graphs;
  while (std::getline(file, line)) {
    sigmo::IntermediateGraph intermediate_graph{line};
    query_graphs.push_back(intermediate_graph.toAMGraph());
  }
  return query_graphs;
}

std::vector<CSRGraph> loadCSRGraphsFromFile(const std::string& filename) {
  std::ifstream file(filename);
  std::string line;
  std::vector<sigmo::CSRGraph> data_graphs;
  while (std::getline(file, line)) {
    sigmo::IntermediateGraph intermediate_graph{line};
    data_graphs.push_back(intermediate_graph.toCSRGraph());
  }
  return data_graphs;
}


} // namespace io
} // namespace sigmo