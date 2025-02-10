#pragma once

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
class GenerateDataSignaturesKernel;

class FilterCandidatesKernel;

} // namespace kernels
} // namespace device
} // namespace mbsm