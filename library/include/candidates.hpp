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

  Candidates(size_t num_nodes, sycl::queue& queue) {
    candidates = sycl::malloc_shared<types::candidates_t>(num_nodes, queue);
    queue.fill(candidates, 0, num_nodes).wait_and_throw();
  }

  void insert(types::node_t candidate) {
    types::candidates_t idx = candidate / (sizeof(types::candidates_t) * 8);
    types::candidates_t offset = candidate % (sizeof(types::candidates_t) * 8);
    candidates[idx] |= (static_cast<types::candidates_t>(1) << offset);
  }

  void remove(types::node_t candidate) {
    types::candidates_t idx = candidate / (sizeof(types::candidates_t) * 8);
    types::candidates_t offset = candidate % (sizeof(types::candidates_t) * 8);
    candidates[idx] &= ~(static_cast<types::candidates_t>(1) << offset);
  }

  void destroy(sycl::queue& queue) { sycl::free(candidates, queue); }
};

} // namespace candidates
} // namespace mbsm