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

  mbsm::candidates::Candidates candidates{device_query_graph.total_nodes, device_data_graph.total_nodes};
  candidates.candidates = sycl::malloc_shared<mbsm::types::candidates_t>(candidates.getAllocationSize(), queue);
  size_t data_nodes = device_data_graph.total_nodes;

  auto e3 = mbsm::isomorphism::filter::filterCandidates(queue, device_query_graph, device_data_graph, query_signatures, data_signatures, candidates);

  e3.wait_and_throw();

  // TODO add assertions

  sycl::free(query_signatures, queue);
  sycl::free(data_signatures, queue);
  sycl::free(candidates.candidates, queue);
  mbsm::destroyDeviceDataGraph(device_data_graph, queue);
  mbsm::destroyDeviceQueryGraph(device_query_graph, queue);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}