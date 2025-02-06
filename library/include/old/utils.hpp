#pragma once

#include "host_data.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace mbsm {
namespace utils {

Graph readGraphFromFile(std::string filename) {
  size_t num_nodes;
  size_t num_edges;

  std::ifstream file(filename);
  file >> num_nodes;
  file >> num_edges;

  std::vector<size_t> row_offsets(num_nodes + 1, 0);
  std::vector<node_t> node_labels(num_nodes, 0);
  std::vector<label_t> col_indices(num_edges, 0);

  for (int i = 0; i < num_nodes; i++) { file >> node_labels[i]; }

  int src, dst;
  for (int i = 0; i < num_edges; i++) {
    file >> src >> dst;
    row_offsets[src + 1]++;
    col_indices[i] = dst;
  }
  file.close();

  for (int i = 1; i < row_offsets.size(); i++) { row_offsets[i] += row_offsets[i - 1]; }

  return Graph{row_offsets, col_indices, node_labels};
}

} // namespace utils
} // namespace mbsm