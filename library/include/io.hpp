#pragma once

#include "pool.hpp"
#include <iostream>
#include <string>

namespace mbsm {
namespace io {

mbsm::GraphPool loadPoolFromBinary(const std::string& filename) {
  mbsm::GraphPool pool;
  // TODO Implement logic to load GraphPool from binary file
  return pool;
}

void savePoolToBinary(const mbsm::GraphPool& pool, const std::string& filename) {
  // TODO Implement logic to save GraphPool to binary file
}

} // namespace io
} // namespace mbsm