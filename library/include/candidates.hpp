/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "device.hpp"
#include "types.hpp"
#include <cstdint>
#include <sycl/sycl.hpp>

namespace sigmo {
namespace candidates {
class Candidates {
public:
  struct CandidatesDevice {
    types::candidates_t* candidates;
    constexpr static types::candidates_t num_bits = sizeof(types::candidates_t) * 8;
    size_t source_nodes;
    size_t target_nodes;
    size_t single_node_size;

    CandidatesDevice() : source_nodes(0), target_nodes(0), single_node_size(0) {}

    CandidatesDevice(size_t source_nodes, size_t target_nodes)
        : source_nodes(source_nodes), target_nodes(target_nodes), single_node_size((target_nodes + (num_bits - 1)) / num_bits) {}

    size_t getAllocationSize() const { return source_nodes * single_node_size * sizeof(types::candidates_t); }

    void setDataCandidates(types::candidates_t* data_candidates) { candidates = data_candidates; }

    SYCL_EXTERNAL void insert(types::node_t source_node, types::node_t candidate) const {
      types::candidates_t idx = candidate / num_bits;
      types::candidates_t offset = candidate % num_bits;
      candidates[source_node * this->single_node_size + idx] |= (static_cast<types::candidates_t>(1) << offset);
    }

    SYCL_EXTERNAL void atomicInsert(types::node_t source_node, types::node_t candidate) const {
      types::candidates_t idx = candidate / num_bits;
      types::candidates_t offset = candidate % num_bits;
      sycl::atomic_ref<types::candidates_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{
          candidates[source_node * this->single_node_size + idx]};
      ref |= (static_cast<types::candidates_t>(1) << offset);
    }

    SYCL_EXTERNAL bool contains(types::node_t source_node, types::node_t candidate) const {
      types::candidates_t idx = candidate / num_bits;
      types::candidates_t offset = candidate % num_bits;
      return (candidates[source_node * this->single_node_size + idx] & (static_cast<types::candidates_t>(1) << offset)) != 0;
    }

    SYCL_EXTERNAL bool atomicContains(types::node_t source_node, types::node_t candidate) const {
      types::candidates_t idx = candidate / num_bits;
      types::candidates_t offset = candidate % num_bits;
      sycl::atomic_ref<types::candidates_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{
          candidates[source_node * this->single_node_size + idx]};
      return (ref & (static_cast<types::candidates_t>(1) << offset)) != 0;
    }

    SYCL_EXTERNAL void remove(types::node_t source_node, types::node_t candidate) const {
      types::candidates_t idx = candidate / num_bits;
      types::candidates_t offset = candidate % num_bits;
      candidates[source_node * this->single_node_size + idx] &= ~(static_cast<types::candidates_t>(1) << offset);
    }

    SYCL_EXTERNAL void atomicRemove(types::node_t source_node, types::node_t candidate) const {
      types::candidates_t idx = candidate / num_bits;
      types::candidates_t offset = candidate % num_bits;
      sycl::atomic_ref<types::candidates_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{
          candidates[source_node * this->single_node_size + idx]};
      ref &= ~(static_cast<types::candidates_t>(1) << offset);
    }

    /**
     * Get the overall number of candidates for a given source node.
     */
    SYCL_EXTERNAL uint32_t getCandidatesCount(types::node_t source_node) const {
      uint32_t count = 0;
      for (size_t i = 0; i < single_node_size; ++i) { count += sycl::popcount(candidates[source_node * single_node_size + i]); }
      return count;
    }

    /**
     * Get the number of candidates for a given source node in a given range.
     */
    SYCL_EXTERNAL uint32_t getCandidatesCount(types::node_t source_node, uint32_t graph_start, uint32_t graph_end) const {
      uint32_t count = 0;
      uint32_t start_idx = graph_start / num_bits;
      uint32_t end_idx = (graph_end + num_bits - 1) / num_bits;

      for (size_t i = start_idx; i < end_idx; ++i) {
        types::candidates_t mask = ~static_cast<types::candidates_t>(0);
        if (start_idx == end_idx - 1) {
          mask = (mask >> (num_bits - (graph_end - graph_start))) << (graph_start % num_bits);
        } else if (i == start_idx) {
          mask <<= (graph_start % num_bits);
        } else if (i == end_idx - 1) {
          mask >>= (num_bits - (graph_end % num_bits));
        }
        count += sycl::popcount(candidates[source_node * single_node_size + i] & mask);
      }

      return count;
    }

    /**
     * Get the candidate in position idx of a query node.
     */
    SYCL_EXTERNAL types::node_t getCandidateAt(types::node_t source_node, uint32_t idx) const {
      uint32_t count = 0;
      for (size_t i = 0; i < single_node_size; ++i) {
        types::candidates_t candidates_block = candidates[source_node * single_node_size + i];
        uint32_t block_count = sycl::popcount(candidates_block);
        if (count + block_count > idx) {
          for (uint32_t bit = 0; bit < num_bits; ++bit) {
            if (candidates_block & (static_cast<types::candidates_t>(1) << bit)) {
              if (count == idx) { return i * num_bits + bit; }
              ++count;
            }
          }
        } else {
          count += block_count;
        }
      }
      // If idx is out of range, return an invalid node ID (assuming -1 is invalid)
      return static_cast<types::node_t>(-1);
    }

    /**
     * Get the candidate in position idx of a query node in a given range.
     */
    SYCL_EXTERNAL types::node_t getCandidateAt(types::node_t source_node, uint32_t idx, uint32_t graph_start, uint32_t graph_end) const {
      uint32_t count = 0;
      uint32_t start_idx = graph_start / num_bits;
      uint32_t end_idx = (graph_end + num_bits - 1) / num_bits;

      for (size_t i = start_idx; i < end_idx; ++i) {
        types::candidates_t mask = ~static_cast<types::candidates_t>(0);
        if (start_idx == end_idx - 1) {
          mask = (mask >> (num_bits - (graph_end - graph_start))) << (graph_start % num_bits);
        } else if (i == start_idx) {
          mask <<= (graph_start % num_bits);
        } else if (i == end_idx - 1) {
          mask >>= (num_bits - (graph_end % num_bits));
        }

        types::candidates_t candidates_block = candidates[source_node * single_node_size + i] & mask;
        uint32_t block_count = sycl::popcount(candidates_block);
        if (count + block_count > idx) {
          while (candidates_block != 0) {
            uint32_t trailing_bits = sycl::ctz(candidates_block);
            if (count == idx) { return i * num_bits + trailing_bits; }
            candidates_block ^= (static_cast<types::candidates_t>(1) << trailing_bits);
            ++count;
          }
        } else {
          count += block_count;
        }
      }
      // If idx is out of range, return an invalid node ID (assuming -1 is invalid)
      return static_cast<types::node_t>(-1);
    }
  }; // struct CandidatesDevice

  Candidates(sycl::queue& queue, size_t source_nodes, size_t target_nodes)
      : queue(queue), candidates(source_nodes, target_nodes), host_candidates(source_nodes, target_nodes) {
    size_t alloc_size = candidates.getAllocationSize();
    candidates.candidates = sycl::malloc_device<types::candidates_t>(alloc_size, queue);
    size_t limit = 4194304;
    sycl::range<1> range(alloc_size < limit ? alloc_size : limit);

    queue
        .submit([&](sycl::handler& cgh) {
          cgh.parallel_for(range, [=, candidates = this->candidates](sycl::item<1> item) {
            for (size_t i = item.get_id(0); i < alloc_size; i += item.get_range(0)) candidates.candidates[i] = 0;
          });
        })
        .wait();
  }
  ~Candidates() {
    sycl::free(candidates.candidates, queue);
    if (host_candidates.candidates != nullptr) delete[] host_candidates.candidates;
  }

  CandidatesDevice getHostCandidates() {
    if (device::memory::default_location == device::memory::MemoryScope::Host) { return candidates; }
    host_candidates.setDataCandidates(new types::candidates_t[candidates.source_nodes * candidates.single_node_size]);
    queue.copy(candidates.candidates, host_candidates.candidates, candidates.source_nodes * candidates.single_node_size).wait();
    return host_candidates;
  }

  size_t getCandidatesCount(types::node_t source_node) const { return candidates.getCandidatesCount(source_node); }
  size_t getCandidatesCount(types::node_t source_node, uint32_t graph_start, uint32_t graph_end) const {
    return candidates.getCandidatesCount(source_node, graph_start, graph_end);
  }
  size_t getAllocationSize() const { return candidates.getAllocationSize(); }
  Candidates::CandidatesDevice getCandidatesDevice() const { return candidates; }

private:
  sycl::queue& queue;
  CandidatesDevice candidates;
  CandidatesDevice host_candidates;

}; // class Candidates
} // namespace candidates
} // namespace sigmo