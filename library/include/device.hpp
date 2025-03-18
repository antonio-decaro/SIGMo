/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "candidates.hpp"
#include <cstdint>
#include <sycl/sycl.hpp>

namespace sigmo {
namespace device {

static struct DeviceOptions {
  size_t join_work_group_size = 128;
  size_t filter_work_group_size = 512;

} deviceOptions;

inline size_t getDeviceMemorySize(sycl::queue& queue) {
  return queue.get_info<sycl::info::queue::device>().get_info<sycl::info::device::global_mem_size>();
}

inline size_t getPreferredWorkGroupSize(sycl::queue& queue) { return queue.get_device().get_info<sycl::info::device::max_work_group_size>(); }

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
class JoinCandidates2Kernel;

} // namespace kernels
} // namespace device
} // namespace sigmo