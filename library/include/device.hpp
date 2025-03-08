/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "candidates.hpp"
#include <cstdint>
#include <sycl/sycl.hpp>

namespace mbsm {
namespace device {

inline size_t getDeviceMemorySize(sycl::queue& queue) {
  return queue.get_info<sycl::info::queue::device>().get_info<sycl::info::device::global_mem_size>();
}

namespace kernels {

class PrefixSumKernel;
class GenerateQuerySignaturesKernel;
class RefineQuerySignaturesKernel;
class GenerateDataSignaturesKernel;
class RefineDataSignaturesKernel;

template<candidates::CandidatesDomain D>
class FilterCandidatesKernel;
template<candidates::CandidatesDomain D>
class RefineCandidatesKernel;
class JoinCandidatesKernel;

} // namespace kernels
} // namespace device
} // namespace mbsm