#pragma once
#include "types.hpp"
#include <cstdint>
#include <sycl/sycl.hpp>

namespace mbsm {
namespace candidates {

template<size_t Bits = 4>
struct Signature {
  uint64_t signature;

  Signature() : signature(0) {}

  Signature(uint64_t signature) : signature(signature) {}

  SYCL_EXTERNAL static uint16_t getMaxLabels() { return sizeof(signature) * 8 / Bits; }

  SYCL_EXTERNAL void setLabelCount(uint8_t label, uint8_t count) {
    if (label < (64 / Bits) && count < (1 << Bits)) {
      signature &= ~((static_cast<uint64_t>((1 << Bits) - 1)) << (label * Bits)); // Clear the bits for the label
      signature |= (static_cast<uint64_t>(count) << (label * Bits));              // Set the new count
    }
  }

  SYCL_EXTERNAL uint8_t getLabelCount(uint8_t label) const {
    if (label < (64 / Bits)) { return (signature >> (label * Bits)) & ((1 << Bits) - 1); }
    return 0;
  }

  SYCL_EXTERNAL void incrementLabelCount(uint8_t label, uint8_t add = 1) {
    if (label < (64 / Bits)) {
      uint8_t count = getLabelCount(label);
      if (count < ((1 << Bits) - 1)) { // Ensure count does not exceed max value
        setLabelCount(label, count + static_cast<uint8_t>(add));
      }
    }
  }
};

struct Candidates {
  types::candidates_t* candidates;
  constexpr static types::candidates_t num_bits = sizeof(types::candidates_t) * 8;
  size_t query_nodes;
  size_t data_nodes;
  size_t single_node_size;

  Candidates() : query_nodes(0), data_nodes(0), single_node_size(0) {}

  Candidates(size_t query_nodes, size_t data_nodes)
      : query_nodes(query_nodes), data_nodes(data_nodes), single_node_size((data_nodes + (num_bits - 1)) / num_bits) {}

  size_t getAllocationSize() const { return query_nodes * single_node_size * sizeof(types::candidates_t); }

  void setDataCandidates(types::candidates_t* data_candidates) { candidates = data_candidates; }

  SYCL_EXTERNAL void insert(types::node_t query_node, types::node_t candidate) const {
    types::candidates_t idx = candidate / num_bits;
    types::candidates_t offset = candidate % num_bits;
    candidates[query_node * this->single_node_size + idx] |= (static_cast<types::candidates_t>(1) << offset);
  }

  SYCL_EXTERNAL void atomicInsert(types::node_t query_node, types::node_t candidate) const {
    types::candidates_t idx = candidate / num_bits;
    types::candidates_t offset = candidate % num_bits;
    sycl::atomic_ref<types::candidates_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{
        candidates[query_node * this->single_node_size + idx]};
    ref |= (static_cast<types::candidates_t>(1) << offset);
  }

  SYCL_EXTERNAL bool contains(types::node_t query_node, types::node_t candidate) const {
    types::candidates_t idx = candidate / num_bits;
    types::candidates_t offset = candidate % num_bits;
    return (candidates[query_node * this->single_node_size + idx] & (static_cast<types::candidates_t>(1) << offset)) != 0;
  }

  SYCL_EXTERNAL bool atomicContains(types::node_t query_node, types::node_t candidate) const {
    types::candidates_t idx = candidate / num_bits;
    types::candidates_t offset = candidate % num_bits;
    sycl::atomic_ref<types::candidates_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{
        candidates[query_node * this->single_node_size + idx]};
    return (ref & (static_cast<types::candidates_t>(1) << offset)) != 0;
  }

  SYCL_EXTERNAL void remove(types::node_t query_node, types::node_t candidate) const {
    types::candidates_t idx = candidate / num_bits;
    types::candidates_t offset = candidate % num_bits;
    candidates[query_node * this->single_node_size + idx] &= ~(static_cast<types::candidates_t>(1) << offset);
  }

  SYCL_EXTERNAL void atomicRemove(types::node_t query_node, types::node_t candidate) const {
    types::candidates_t idx = candidate / num_bits;
    types::candidates_t offset = candidate % num_bits;
    sycl::atomic_ref<types::candidates_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{
        candidates[query_node * this->single_node_size + idx]};
    ref &= ~(static_cast<types::candidates_t>(1) << offset);
  }

  SYCL_EXTERNAL uint32_t getCandidatesCount(types::node_t query_node) const {
    uint32_t count = 0;
    for (size_t i = 0; i < single_node_size; ++i) { count += sycl::popcount(candidates[query_node * single_node_size + i]); }
    return count;
  }
};

} // namespace candidates
} // namespace mbsm