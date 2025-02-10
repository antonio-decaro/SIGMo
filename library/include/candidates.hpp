#pragma once
#include "types.hpp"
#include <cstdint>
#include <sycl/sycl.hpp>

namespace mbsm {
namespace candidates {

struct Signature {
  uint32_t signature;

  Signature() : signature(0) {}

  Signature(uint32_t signature) : signature(signature) {}

  SYCL_EXTERNAL void setLabelCount(uint8_t label, uint8_t count) {
    if (label < 16 && count < 4) {
      signature &= ~(0x3 << (label * 2));  // Clear the bits for the label
      signature |= (count << (label * 2)); // Set the new count
    }
  }

  SYCL_EXTERNAL uint8_t getLabelCount(uint8_t label) const {
    if (label < 16) { return (signature >> (label * 2)) & 0x3; }
    return 0;
  }

  SYCL_EXTERNAL void incrementLabelCount(uint8_t label) {
    if (label < 16) {
      uint8_t count = getLabelCount(label);
      if (count < 3) { // Ensure count does not exceed 3
        setLabelCount(label, count + static_cast<uint8_t>(1));
      }
    }
  }
};

struct Candidates {
  types::candidates_t* candidates;
  constexpr static types::candidates_t num_bits = sizeof(types::candidates_t) * 8;

  Candidates(size_t num_nodes, sycl::queue& queue) {
    size_t size = (num_nodes + (num_bits - 1)) / num_bits;
    candidates = sycl::malloc_shared<types::candidates_t>(size, queue);
    queue.fill(candidates, static_cast<types::candidates_t>(0), size).wait_and_throw();
  }

  void insert(types::node_t candidate) {
    types::candidates_t idx = candidate / num_bits;
    types::candidates_t offset = candidate % num_bits;
    candidates[idx] |= (static_cast<types::candidates_t>(1) << offset);
  }

  void remove(types::node_t candidate) {
    types::candidates_t idx = candidate / num_bits;
    types::candidates_t offset = candidate % num_bits;
    candidates[idx] &= ~(static_cast<types::candidates_t>(1) << offset);
  }

  uint32_t getCandidatesCount(size_t num_nodes) const {
    uint32_t count = 0;
    size_t size = (num_nodes + (num_bits - 1)) / num_bits;
    for (size_t i = 0; i < size; ++i) { count += sycl::popcount(candidates[i]); }
    return count;
  }

  void destroy(sycl::queue& queue) { sycl::free(candidates, queue); }
};

} // namespace candidates
} // namespace mbsm