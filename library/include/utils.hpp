#pragma once

#include "types.hpp"
#include <chrono>
#include <cmath>
#include <sycl/sycl.hpp>
#include <vector>

namespace mbsm {
namespace utils {

template<typename TypeT = types::adjacency_t>
SYCL_EXTERNAL uint8_t getNumOfAdjacencyIntegers(uint8_t num_nodes) {
  uint8_t num_bits = sizeof(TypeT) * 8;
  return (num_nodes * num_nodes + num_bits - 1) / num_bits;
}

namespace adjacency_matrix {
template<typename TypeT>
SYCL_EXTERNAL void setBit(TypeT* adjacency_matrix, uint8_t adjacency_matrix_size, uint8_t u, uint8_t v) {
  uint16_t num_bits = sizeof(TypeT) * 8;
  uint16_t num_nodes = sycl::sqrt(static_cast<float>(num_bits * adjacency_matrix_size));
  uint16_t idx_u = (u * num_nodes + v);
  uint16_t idx_v = (v * num_nodes + u);
  adjacency_matrix[idx_u / num_bits] |= static_cast<TypeT>(1) << (idx_u % num_bits);
  adjacency_matrix[idx_v / num_bits] |= static_cast<TypeT>(1) << (idx_v % num_bits);
}

template<typename TypeT>
SYCL_EXTERNAL void getNeighbors(TypeT* adjacency_matrix, uint8_t adjacency_matrix_size, uint8_t u, types::node_t* neighbors) {
  uint16_t num_bits = sizeof(TypeT) * 8;
  uint16_t num_nodes = sycl::sqrt(static_cast<float>(num_bits * adjacency_matrix_size));
  uint16_t neighbor_count = 0;

  for (uint8_t v = 0; v < num_nodes; ++v) {
    uint16_t idx = (u * num_nodes + v);
    if (adjacency_matrix[idx / num_bits] & static_cast<TypeT>(static_cast<TypeT>(1) << (idx % num_bits))) { neighbors[neighbor_count++] = v; }
  }
  neighbors[neighbor_count] = types::NULL_NODE; // Null-terminate the list of neighbors
}

} // namespace adjacency_matrix

SYCL_EXTERNAL uint32_t binarySearch(const uint32_t* num_nodes, uint32_t total_graphs, uint32_t node_id) {
  uint32_t low = 0;
  uint32_t high = total_graphs - 1;
  while (low < high) {
    uint32_t mid = (low + high) / 2;
    if (num_nodes[mid] <= node_id) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

class BatchedEvent {
public:
  BatchedEvent() = default;

  void add(sycl::event e) { events.push_back(e); }

  void wait() {
    for (auto& e : events) { e.wait(); }
  }

  void clear() { events.clear(); }

  void addAndClear(sycl::event e) {
    wait();
    clear();
    add(e);
  }

  sycl::event getLastEvent() { return events.back(); }

  std::chrono::duration<double> getProfilingInfo() {
    std::chrono::nanoseconds total_time(0);
    for (auto& e : events) {
      auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
      total_time += std::chrono::nanoseconds(end - start);
    }
    return std::chrono::duration_cast<std::chrono::duration<double>>(total_time);
  }

private:
  std::vector<sycl::event> events;
};

} // namespace utils
} // namespace mbsm