#pragma once

#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include "types.hpp"
#include "mapping.hpp"

namespace msm {

/**
 * @brief Represents a graph data structure implemented as a CSR matrix.
 */
class Graph {
protected:
  std::vector<size_t> row_offsets; /**< The row offsets of the graph. */
  std::vector<node_t> col_indices; /**< The column indices of the graph. */
  std::vector<label_t> node_labels; /**< The labels of the nodes in the graph. */

public:
  /**
   * @brief Default constructor for Graph.
   */
  Graph() {
    row_offsets.push_back(0);
  }

  /**
   * @brief Constructor for Graph.
   * @param row_offsets The row offsets of the graph.
   * @param col_indices The column indices of the graph.
   * @param node_labels The labels of the nodes in the graph.
   */
  Graph(std::vector<size_t> row_offsets, std::vector<node_t> col_indices, std::vector<label_t> node_labels) :
    row_offsets(row_offsets), col_indices(col_indices), node_labels(node_labels) {};

  /**
   * @brief Copy constructor for Graph.
   * @param other The Graph object to be copied.
   */
  Graph(const Graph& other) = default;

  /**
   * @brief Move constructor for Graph.
   * @param other The Graph object to be moved.
   */
  Graph(Graph&& other) {
    row_offsets = std::move(other.row_offsets);
    col_indices = std::move(other.col_indices);
    node_labels = std::move(other.node_labels);
  }

  /**
   * @brief Get the number of nodes in the graph.
   * @return The number of nodes in the graph.
   */
  const size_t getNumNodes() const { return row_offsets.size() - 1; }

  /**
   * @brief Get the row offsets of the graph.
   * @return The row offsets of the graph.
   */
  const std::vector<size_t>& getRowOffsets() const { return row_offsets; }

  /**
   * @brief Get the column indices of the graph.
   * @return The column indices of the graph.
   */
  const std::vector<node_t>& getColIndices() const { return col_indices; }

  /**
   * @brief Get the node labels of the graph.
   * @return The node labels of the graph.
   */
  const std::vector<label_t>& getNodeLabels() const { return node_labels; }

  /**
   * @brief Check if two nodes are neighbors in the graph.
   * @param src The source node.
   * @param dst The destination node.
   * @return True if the nodes are neighbors, false otherwise.
   */
  const bool isNeighbour(node_t src, node_t dst) const {
    auto begin = col_indices.begin() + row_offsets[src];
    auto end = col_indices.begin() + row_offsets[src + 1];
    
    return std::find(begin, end, dst) != end;
  }

  /**
   * @brief Get the neighbors of a node in the graph.
   * @param node The node for which to get the neighbors.
   * @return A vector containing the neighbors of the node.
   */
  const std::vector<node_t> getNeighbours(node_t node) const {
    return std::vector<node_t>(col_indices.begin() + row_offsets[node], col_indices.begin() + row_offsets[node + 1]);
  }

  /**
   * @brief Add a new node to the graph.
   * @param label The label of the new node.
   */
  void addNode(label_t label) noexcept {
    row_offsets.push_back(row_offsets.back());
    node_labels.push_back(label);
  }

  /**
   * @brief Add an edge between two nodes in the graph.
   * @param src The source node.
   * @param dst The destination node.
   * @throws std::runtime_error if the node index is out of bounds or the edge already exists.
   */
  void addEdge(node_t src, node_t dst) {
    if (src >= getNumNodes() || dst >= getNumNodes()) {
      throw std::runtime_error("Node index out of bounds");
    }

    auto begin = col_indices.begin() + row_offsets[src];
    auto end = col_indices.begin() + row_offsets[src + 1];

    if (std::find(begin, end, dst) != end) {
      throw std::runtime_error("Edge already exists");
    }

    col_indices.insert(end, dst);
    for (size_t i = src + 1; i < row_offsets.size(); i++) {
      row_offsets[i]++;
    }

    row_offsets[src + 1]++;
  }

  /**
   * @brief Set the label of a node in the graph.
   * @param node The node for which to set the label.
   * @param label The new label for the node.
   * @throws std::runtime_error if the node index is out of bounds.
   */
  void setLabel(node_t node, label_t label) {
    if (node >= getNumNodes()) {
      throw std::runtime_error("Node index out of bounds");
    }

    node_labels[node] = label;
  }

  /**
   * @brief Get the label of a node in the graph.
   * @param node The node for which to get the label.
   * @return The label of the node.
   * @throws std::runtime_error if the node index is out of bounds.
   */
  const label_t getLabel(node_t node) const { return node_labels[node]; }
};

/**
 * @brief Represents an undirected graph.
 * 
 * This class extends the Graph class and provides additional functionality for undirected graphs.
 */
class UnGraph : public Graph {
public:
  /**
   * @brief Move constructor for UnGraph.
   * 
   * @param other The UnGraph object to be moved.
   */
  UnGraph(UnGraph&& other) : Graph(std::move(other)) {}

  /**
   * @brief Copy constructor for UnGraph.
   * 
   * @param other The Graph object to be copied.
   */
  UnGraph(const Graph& other) : Graph(other) {
    for (node_t node = 0; node < getNumNodes(); node++) {
      for (auto neighbour : getNeighbours(node)) {
        if (!isNeighbour(neighbour, node)) {
          addEdge(neighbour, node);
        }
      }
    }
  }

  /**
   * @brief Constructor for UnGraph.
   * 
   * @param row_offsets The row offsets of the graph.
   * @param col_indices The column indices of the graph.
   * @param node_labels The labels of the nodes in the graph.
   */
  UnGraph(std::vector<size_t> row_offsets, std::vector<node_t> col_indices, std::vector<label_t> node_labels) :
    Graph(row_offsets, col_indices, node_labels) {
    for (node_t node = 0; node < getNumNodes(); node++) {
      for (auto neighbour : getNeighbours(node)) {
        if (!isNeighbour(neighbour, node)) {
          addEdge(neighbour, node);
        }
      }
    }
  }

  /**
   * @brief Adds an edge between two nodes in the graph.
   * 
   * This function adds an edge between the source node and the destination node,
   * and also adds an edge between the destination node and the source node.
   * 
   * @param src The source node.
   * @param dst The destination node.
   */
  void addEdge(node_t src, node_t dst) {
    Graph::addEdge(src, dst);
    Graph::addEdge(dst, src);
  }
};

class CompressedGraphs {
private:
  std::vector<Graph>& graphs;
  std::vector<size_t> row_offsets;
  std::vector<node_t> col_indices;
  std::vector<label_t> node_labels;
  std::vector<size_t> offsets;
  std::vector<size_t> sizes;

public:
  CompressedGraphs(std::vector<Graph>& graphs) : graphs(graphs) {
    size_t offset = 0;
    size_t size = 0;
    size_t row_size = 0;

    row_offsets.push_back(0);
    for (auto& graph : graphs) {
      offsets.push_back(offset);

      for (auto it = graph.getRowOffsets().begin() + 1; it < graph.getRowOffsets().end(); it++) {
        row_offsets.push_back(*it + row_size);
      }
      col_indices.insert(col_indices.end(), graph.getColIndices().begin(), graph.getColIndices().end());
      node_labels.insert(node_labels.end(), graph.getNodeLabels().begin(), graph.getNodeLabels().end());

      size = graph.getNumNodes();
      sizes.push_back(size);
      row_size += graph.getRowOffsets().back();
      offset += size;
    }
  }

  const std::vector<size_t>& getRowOffsets() const { return row_offsets; }
  const std::vector<node_t>& getColIndices() const { return col_indices; }
  const std::vector<label_t>& getNodeLabels() const { return node_labels; }
  const std::vector<size_t>& getOffsets() const { return offsets; }
  const std::vector<size_t>& getSizes() const { return sizes; }
  const size_t getNumGraphs() const { return graphs.size(); }
  const std::vector<Graph>& getGraphs() const { return graphs; }
};


} // namespace msm
