#pragma once

#include "types.hpp"
#include <sycl/sycl.hpp>
#include <vector>

namespace mbsm {
namespace detail {
class BitmaskMap {
private:
  const size_t source_size;
  const size_t target_size;
  const size_t single_node_mask_lenght;

protected:
  sycl::buffer<mask_t, 1> masks_buffer;
  // std::vector<mask_t> masks;

public:
  BitmaskMap(size_t source_size, size_t target_size, bool init = false) noexcept
      : source_size(source_size), target_size(target_size), single_node_mask_lenght(target_size / MASK_SIZE + 1),
        masks_buffer(sycl::range{source_size * single_node_mask_lenght}) {
    if (init) {
      setOffMask();
      setOnMask();
    }
  }

  inline const size_t getSourceSize() const noexcept { return source_size; }

  inline const size_t getTargetSize() const noexcept { return target_size; }

  inline const size_t getSingleMaskLenght() const noexcept { return single_node_mask_lenght; }

  inline bool get(node_t source_node, node_t target_node, sycl::host_accessor<mask_t, 1>& masks) {
    return masks[(source_node * single_node_mask_lenght) + target_node / MASK_SIZE] & (((mask_t)1) << (target_node % MASK_SIZE));
  }

  inline bool get(node_t source_node, node_t target_node) {
    sycl::host_accessor masks{masks_buffer};
    return get(source_node, target_node, masks);
  }

  inline void setOff(node_t source_node, node_t target_node, sycl::host_accessor<mask_t, 1>& masks) {
    masks[(source_node * single_node_mask_lenght) + target_node / MASK_SIZE] &= ~(((mask_t)1) << (target_node % MASK_SIZE));
  }

  inline void setOff(node_t source_node, node_t target_node) {
    sycl::host_accessor masks{masks_buffer};
    setOff(source_node, target_node, masks);
  }

  inline void setOn(node_t source_node, node_t target_node, sycl::host_accessor<mask_t, 1>& masks) {
    masks[(source_node * single_node_mask_lenght) + target_node / MASK_SIZE] |= (((mask_t)1) << (target_node % MASK_SIZE));
  }

  inline void setOn(node_t source_node, node_t target_node) {
    sycl::host_accessor masks{masks_buffer};
    setOn(source_node, target_node, masks);
  }

  void setOffMask() noexcept {
    sycl::host_accessor masks{masks_buffer};
    for (int i = 0; i < masks.size(); i++) { masks[i] = 0; }
  }

  void setOnMask() noexcept {
    for (int i = 0; i < source_size; i++) {
      for (int j = 0; j < target_size; j++) { setOn(i, j); }
    }
  }

  [[deprecated("Use getOnNodes instead")]]
  const std::vector<mask_t> operator[](node_t source_node) {
    sycl::host_accessor masks{masks_buffer};
    auto first = masks.begin() + source_node * single_node_mask_lenght;
    auto last = masks.begin() + (source_node + 1) * single_node_mask_lenght;

    return std::vector<mask_t>{first, last};
  }

  std::vector<node_t> getOnNodes(node_t source_node) {
    sycl::host_accessor masks{masks_buffer};
    std::vector<node_t> on_nodes;
    auto offset = source_node * single_node_mask_lenght;
    for (int i = 0; i < single_node_mask_lenght; i++) {
      if (masks[offset + i] != 0) {
        for (int j = 0; j < MASK_SIZE && (i * MASK_SIZE + j) < target_size; j++) {
          if (get(source_node, i * MASK_SIZE + j, masks)) { on_nodes.push_back(i * MASK_SIZE + j); }
        }
      }
    }
    return on_nodes;
  }

  sycl::accessor<mask_t, 1, sycl::access::mode::read_write> getDeviceAccessor(sycl::handler& cgh) {
    return sycl::accessor{masks_buffer, cgh, sycl::read_write};
  }

  sycl::host_accessor<mask_t, 1> getHostAccessor() { return sycl::host_accessor{masks_buffer}; }

  size_t size() { return source_size * single_node_mask_lenght; }
};
} // namespace detail
} // namespace mbsm
