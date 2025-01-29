#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace msm {
typedef unsigned short node_t;
typedef unsigned short label_t;
constexpr label_t WILDCARD = 0;

typedef struct {
  std::string name;
  double join_time;
  double filter_time;
} bench_t;

typedef uint32_t mask_t;
constexpr size_t MASK_SIZE = sizeof(mask_t) * 8;

}

