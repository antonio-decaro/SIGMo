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

namespace mbsm {
namespace io {

std::vector<AMGraph> loadQueryGraphsFromFile(const std::string& filename) {
  std::ifstream file(filename);
  std::string line;
  std::vector<mbsm::AMGraph> query_graphs;
  while (std::getline(file, line)) {
    mbsm::IntermediateGraph intermediate_graph{line};
    query_graphs.push_back(intermediate_graph.toAMGraph());
  }
  return query_graphs;
}

std::vector<CSRGraph> loadDataGraphsFromFile(const std::string& filename) {
  std::ifstream file(filename);
  std::string line;
  std::vector<mbsm::CSRGraph> data_graphs;
  while (std::getline(file, line)) {
    mbsm::IntermediateGraph intermediate_graph{line};
    data_graphs.push_back(intermediate_graph.toCSRGraph());
  }
  return data_graphs;
}

mbsm::GraphPool loadPoolFromBinary(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) { throw std::runtime_error("Unable to open file for reading: " + filename); }

  mbsm::GraphPool pool;

  // Read the number of data graphs
  uint32_t num_data_graphs;
  file.read(reinterpret_cast<char*>(&num_data_graphs), sizeof(num_data_graphs));

  // Read each data graph
  for (uint32_t i = 0; i < num_data_graphs; ++i) {
    // Read the number of nodes
    uint32_t num_nodes;
    file.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));

    // Read the row offsets
    std::vector<types::row_offset_t> row_offsets(num_nodes + 1);
    file.read(reinterpret_cast<char*>(row_offsets.data()), (num_nodes + 1) * sizeof(types::row_offset_t));

    // Read the column indices
    uint32_t num_edges = row_offsets[num_nodes];
    std::vector<types::col_index_t> column_indices(num_edges);
    file.read(reinterpret_cast<char*>(column_indices.data()), num_edges * sizeof(types::col_index_t));

    // Read the labels
    std::vector<types::label_t> labels(num_nodes);
    file.read(reinterpret_cast<char*>(labels.data()), num_nodes * sizeof(types::label_t));

    // Create the CSRGraph and add it to the pool
    pool.getDataGraphs().emplace_back(row_offsets, column_indices, labels, num_nodes);
  }

  // Read the number of query graphs
  uint32_t num_query_graphs;
  file.read(reinterpret_cast<char*>(&num_query_graphs), sizeof(num_query_graphs));

  // Read each query graph
  for (uint32_t i = 0; i < num_query_graphs; ++i) {
    // Read the number of nodes
    uint32_t num_nodes;
    file.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));

    // Read the adjacency matrix
    uint32_t adjacency_size = mbsm::utils::getNumOfAdjacencyIntegers(num_nodes);
    std::vector<types::adjacency_t> adjacency(adjacency_size);
    file.read(reinterpret_cast<char*>(adjacency.data()), adjacency_size * sizeof(types::adjacency_t));

    // Read the labels
    std::vector<types::label_t> labels(num_nodes);
    file.read(reinterpret_cast<char*>(labels.data()), num_nodes * sizeof(types::label_t));

    // Create the AMGraph and add it to the pool
    pool.getQueryGraphs().emplace_back(adjacency, labels, num_nodes);
  }

  file.close();
  return pool;
}

void savePoolToBinary(mbsm::GraphPool& pool, const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing: " + filename); }

  // Write the number of data graphs
  uint32_t num_data_graphs = pool.getDataGraphs().size();
  file.write(reinterpret_cast<const char*>(&num_data_graphs), sizeof(num_data_graphs));

  // Write each data graph
  for (auto& graph : pool.getDataGraphs()) {
    // Write the number of nodes
    uint32_t num_nodes = graph.getNumNodes();
    file.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));

    // Write the row offsets
    file.write(reinterpret_cast<const char*>(graph.getRowOffsets()), (num_nodes + 1) * sizeof(types::row_offset_t));

    // Write the column indices
    uint32_t num_edges = graph.getRowOffsets()[num_nodes];
    file.write(reinterpret_cast<const char*>(graph.getColumnIndices()), num_edges * sizeof(types::col_index_t));

    // Write the labels
    file.write(reinterpret_cast<const char*>(graph.getLabels()), num_nodes * sizeof(types::label_t));
  }

  // Write the number of query graphs
  uint32_t num_query_graphs = pool.getQueryGraphs().size();
  file.write(reinterpret_cast<const char*>(&num_query_graphs), sizeof(num_query_graphs));

  // Write each query graph
  for (auto& graph : pool.getQueryGraphs()) {
    // Write the number of nodes
    uint32_t num_nodes = graph.getNumNodes();
    file.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));

    // Write the adjacency matrix
    uint32_t adjacency_size = mbsm::utils::getNumOfAdjacencyIntegers(num_nodes);
    file.write(reinterpret_cast<const char*>(graph.getAdjacencyMatrix()), adjacency_size * sizeof(types::adjacency_t));

    // Write the labels
    file.write(reinterpret_cast<const char*>(graph.getLabels()), num_nodes * sizeof(types::label_t));
  }

  file.close();
}

} // namespace io
} // namespace mbsm