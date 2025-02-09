#include "./include/data.hpp"
#include "./include/utils.hpp"
#include "gtest/gtest.h"
#include <bitset>
#include <mbsm.hpp>
#include <sycl/sycl.hpp>

TEST(FilterTest, FilterTest) {
  auto pool = mbsm::io::loadPoolFromBinary(TEST_POOL_PATH);

  sycl::queue queue{sycl::gpu_selector_v};

  auto device_query_graph = mbsm::createDeviceQueryGraph(queue, pool.getQueryGraphs());
  auto device_data_graph = mbsm::createDeviceDataGraph(queue, pool.getDataGraphs());

  mbsm::candidates::Signature* query_signatures = sycl::malloc_shared<mbsm::candidates::Signature>(device_query_graph.total_nodes, queue);
  mbsm::candidates::Signature* data_signatures = sycl::malloc_shared<mbsm::candidates::Signature>(device_data_graph.total_nodes, queue);

  auto e1 = mbsm::isomorphism::filter::generateQuerySignatures(queue, device_query_graph, query_signatures);
  auto e2 = mbsm::isomorphism::filter::generateDataSignatures(queue, device_data_graph, data_signatures);

  queue.wait_and_throw();

  // TODO: Add wait_and_throw() to the end of the test

  sycl::free(query_signatures, queue);
  sycl::free(data_signatures, queue);
  mbsm::destroyDeviceDataGraph(device_data_graph, queue);
  mbsm::destroyDeviceQueryGraph(device_query_graph, queue);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}