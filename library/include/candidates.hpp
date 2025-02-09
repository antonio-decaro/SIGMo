#pragma once
#include <cstdint>
#include <sycl/sycl.hpp>

namespace mbsm {
namespace candidates {

struct Signature {
  uint32_t signature;

  Signature() : signature(0) {}

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

} // namespace candidates
} // namespace mbsm