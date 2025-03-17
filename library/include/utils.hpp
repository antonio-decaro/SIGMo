/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "types.hpp"
#include <chrono>
#include <cmath>
#include <sycl/sycl.hpp>
#include <vector>

namespace mbsm {
namespace utils {

template<typename TypeT = types::adjacency_t>
SYCL_EXTERNAL inline uint8_t getNumOfAdjacencyIntegers(uint8_t num_nodes) {
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
SYCL_EXTERNAL void getNeighbors(
    TypeT* adjacency_matrix, uint8_t adjacency_matrix_size, types::node_t node_id, types::node_t* neighbors, types::node_t previous_nodes = 0) {
  uint8_t u = node_id - previous_nodes;
  uint16_t num_bits = sizeof(TypeT) * 8;
  uint16_t num_nodes = sycl::sqrt(static_cast<float>(num_bits * adjacency_matrix_size));
  uint16_t neighbor_count = 0;

  for (uint8_t v = 0; v < num_nodes; ++v) {
    uint16_t idx = (u * num_nodes + v);
    if (adjacency_matrix[idx / num_bits] & static_cast<TypeT>(static_cast<TypeT>(1) << (idx % num_bits))) {
      neighbors[neighbor_count++] = v + previous_nodes;
    }
  }
  if (neighbor_count < types::MAX_NEIGHBORS) neighbors[neighbor_count] = types::NULL_NODE; // Null-terminate the list of neighbors
}

template<typename TypeT>
SYCL_EXTERNAL bool isNeighbor(
    TypeT* adjacency_matrix, uint8_t adjacency_matrix_size, types::node_t node_id, types::node_t neighbor_id, types::node_t previous_nodes = 0) {
  uint8_t u = node_id - previous_nodes;
  uint8_t v = neighbor_id - previous_nodes;
  uint16_t num_bits = sizeof(TypeT) * 8;
  uint16_t num_nodes = sycl::sqrt(static_cast<float>(num_bits * adjacency_matrix_size));
  uint16_t idx = u * num_nodes + v;
  return adjacency_matrix[idx / num_bits] & static_cast<TypeT>(static_cast<TypeT>(1) << (idx % num_bits));
}

} // namespace adjacency_matrix

namespace detail {
template<typename TypeT>
struct Bitset {
  TypeT data;
  uint offset;
  Bitset() : data(0), offset(0) {}
  Bitset(uint offset) : data(0), offset(offset) {}
  SYCL_EXTERNAL inline void set(uint8_t idx) { data |= static_cast<TypeT>(1) << (idx - offset); }
  SYCL_EXTERNAL inline void unset(uint8_t idx) { data &= ~(static_cast<TypeT>(1) << (idx - offset)); }
  SYCL_EXTERNAL inline bool get(uint8_t idx) { return data & (static_cast<TypeT>(1) << (idx - offset)); }
  SYCL_EXTERNAL inline void clear() { data = 0; }
  SYCL_EXTERNAL inline void merge(const Bitset& other) { data |= other.data; }
  SYCL_EXTERNAL inline void intersection(const Bitset& other) { data &= other.data; }
  SYCL_EXTERNAL inline Bitset& operator=(const Bitset& other) {
    data = other.data;
    return *this;
  }
  SYCL_EXTERNAL inline bool operator==(const Bitset& other) { return data == other.data; }
  SYCL_EXTERNAL inline bool empty() { return data == 0; }
  SYCL_EXTERNAL inline uint16_t size() { return sycl::popcount(data); }
  SYCL_EXTERNAL inline uint16_t getSetBit(uint8_t idx) {
    idx -= offset;
    uint count = 0;
    uint16_t ret = -1;
    TypeT tmp = data;
    for (int count = 0; count <= idx; ++count) {
      ret = sycl::ctz(tmp);
      tmp &= ~(static_cast<TypeT>(1) << ret);
    }
    return ret;
  }
};

} // namespace detail

SYCL_EXTERNAL uint32_t binaryAMSearch(const uint32_t* num_nodes, uint32_t total_graphs, uint32_t node_id) {
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

SYCL_EXTERNAL uint32_t binarySearch(const uint32_t* vector, uint32_t size, uint32_t value) {
  size_t lo = 0, hi = size;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (value >= vector[mid]) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return (lo > 0) ? (lo - 1) : static_cast<uint32_t>(-1);
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

  size_t numEvents() { return events.size(); }

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