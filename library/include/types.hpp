#pragma once

#include <cstdint>
#include <limits>

namespace mbsm {
namespace types {

using row_offset_t = uint32_t;
using col_index_t = uint32_t;
using label_t = uint8_t;
using node_t = uint32_t;
using mask_t = uint64_t;
using adjacency_t = uint64_t;
using candidates_t = uint64_t;

// define null
constexpr node_t NULL_NODE = std::numeric_limits<node_t>::max();

constexpr std::size_t MAX_NEIGHBORS = 4;


} // namespace types
} // namespace mbsm