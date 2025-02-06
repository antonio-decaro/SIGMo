#pragma once

#include "types.hpp"
#include <sycl/sycl.hpp>

namespace mbsm {
namespace device {

namespace bitmask {

extern SYCL_EXTERNAL inline sycl::atomic_ref<mask_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group>
getAtomicRef(sycl::accessor<mask_t, 1> masks, node_t source_node, node_t target_node, size_t single_node_mask_size, size_t bitmask_size) {
  return sycl::atomic_ref<mask_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group>(
      masks[(single_node_mask_size * source_node) + target_node / bitmask_size]);
}

extern SYCL_EXTERNAL inline void
setOn(sycl::accessor<mask_t, 1> masks, node_t source_node, node_t target_node, size_t single_node_mask_size, size_t bitmask_size) {
  masks[(source_node * single_node_mask_size) + target_node / bitmask_size] ^= ((mask_t)1 << (target_node % bitmask_size));
}

extern SYCL_EXTERNAL inline void
setOn(sycl::atomic_ref<mask_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> mask, node_t target_node, size_t bitmask_size) {
  mask ^= ((mask_t)1 << (target_node % bitmask_size));
}

extern SYCL_EXTERNAL inline bool
get(sycl::atomic_ref<mask_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> mask, node_t target_node, size_t bitmask_size) {
  return mask & ((mask_t)1 << (target_node % bitmask_size));
}

extern SYCL_EXTERNAL inline bool
get(sycl::accessor<mask_t, 1> masks, node_t source_node, node_t target_node, size_t single_node_mask_size, size_t bitmask_size) {
  return masks[(source_node * single_node_mask_size) + target_node / bitmask_size] & ((mask_t)1 << (target_node % bitmask_size));
}

extern SYCL_EXTERNAL inline void
setOff(sycl::accessor<mask_t, 1> masks, node_t source_node, node_t target_node, size_t single_node_mask_size, size_t bitmask_size) {
  masks[(source_node * single_node_mask_size) + target_node / bitmask_size] &= ~((mask_t)1 << (target_node % bitmask_size));
}

extern SYCL_EXTERNAL inline void
setOff(sycl::atomic_ref<mask_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> mask, node_t target_node, size_t bitmask_size) {
  mask &= ~((mask_t)1 << (target_node % bitmask_size));
}

extern SYCL_EXTERNAL inline size_t getNumCandidates(
    const sycl::accessor<mask_t, 1>& candidates, node_t source_node, size_t data_nodes, size_t single_node_mask_size, size_t bitmask_size) {
  size_t num_candidates = 0;
  for (int i = 0; i < data_nodes; i++) {
    if (bitmask::get(candidates, source_node, i, single_node_mask_size, bitmask_size)) { num_candidates++; }
  }
  return num_candidates;
}

} // namespace bitmask

namespace set {} // namespace set

namespace graph {

extern SYCL_EXTERNAL inline bool labelMatch(mbsm::label_t label1, mbsm::label_t label2, bool use_wildcard = false) {
  return label1 == label2 || (use_wildcard && (label2 == WILDCARD || label1 == WILDCARD));
}

extern SYCL_EXTERNAL inline bool isNeighbour(node_t source_node,
                                             node_t target_node,
                                             const sycl::accessor<size_t, 1, sycl::access::mode::read>& row_offset,
                                             const sycl::accessor<node_t, 1, sycl::access::mode::read>& col_indices) {
  for (int i = row_offset[source_node]; i < row_offset[source_node + 1]; i++) {
    if (col_indices[i] == target_node) { return true; }
  }
  return false;
}

extern SYCL_EXTERNAL inline bool isIsomorphic(node_t query_node,
                                              node_t data_node,
                                              size_t offset,
                                              const sycl::accessor<size_t, 1, sycl::access::mode::read>& data_row_offset,
                                              const sycl::accessor<size_t, 1, sycl::access::mode::read>& query_row_offset,
                                              const sycl::accessor<node_t, 1, sycl::access::mode::read>& data_col_indices,
                                              const sycl::accessor<node_t, 1, sycl::access::mode::read>& query_col_indices,
                                              const sycl::accessor<label_t, 1, sycl::access::mode::read>& data_labels,
                                              const sycl::accessor<label_t, 1, sycl::access::mode::read>& query_labels,
                                              sycl::accessor<mask_t, 1> masks,
                                              size_t single_node_mask_size,
                                              size_t bitmask_size) {
  for (int i = query_row_offset[offset + query_node]; i < query_row_offset[offset + query_node + 1]; i++) {
    node_t query_neighbour = offset + query_col_indices[i];
    bool found = false;
    for (int j = data_row_offset[data_node]; j < data_row_offset[data_node + 1]; j++) {
      node_t data_neighbour = data_col_indices[j];
      if (bitmask::get(masks, query_neighbour, data_neighbour, single_node_mask_size, bitmask_size)
          && mbsm::device::graph::labelMatch(query_labels[query_neighbour], data_labels[data_neighbour])) {
        found = true;
        break;
      }
    }
    if (!found) { return false; }
  }
  return true;
}

extern SYCL_EXTERNAL void
computeNextFrontier(size_t node_begin,
                    size_t node_end,
                    const sycl::_V1::accessor<mbsm::node_t, 1, sycl::_V1::access::mode::read>& query_col_indices,
                    const sycl::_V1::accessor<bool, 1, sycl::_V1::access::mode::read_write>& visited,
                    const sycl::_V1::local_accessor<mbsm::node_t, 1>& next_frontier,
                    sycl::_V1::atomic_ref<size_t, sycl::_V1::memory_order::relaxed, sycl::_V1::memory_scope::sub_group>& next_frontier_size_ref) {
  for (int i = node_begin; i < node_end; i++) {
    auto neighbour = query_col_indices[i];
    if (!visited[neighbour]) {
      next_frontier[next_frontier_size_ref.fetch_add(1)] = neighbour;
      visited[neighbour] = true;
    }
  }
}

} // namespace graph

} // namespace device

} // namespace mbsm