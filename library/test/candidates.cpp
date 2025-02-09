#include "./include/data.hpp"
#include "./include/utils.hpp"
#include "gtest/gtest.h"
#include <bitset>
#include <mbsm.hpp>
#include <sycl/sycl.hpp>

TEST(SignatureTest, CheckSignatureMethods) {
  mbsm::candidates::Signature signature;
  uint64_t expected_signature = 0b01000000000000000000100000110001;

  signature.setLabelCount(0, 1);
  signature.setLabelCount(2, 3);
  signature.setLabelCount(5, 2);
  signature.setLabelCount(15, 1);

  ASSERT_EQ(signature.signature, expected_signature);

  ASSERT_EQ(signature.getLabelCount(15), static_cast<uint8_t>(1));

  signature.incrementLabelCount(15);
  expected_signature = 0b10000000000000000000100000110001;

  ASSERT_EQ(signature.signature, expected_signature);

  signature.incrementLabelCount(14);
  expected_signature = 0b10010000000000000000100000110001;

  ASSERT_EQ(signature.signature, expected_signature);

  signature.incrementLabelCount(2);
  ASSERT_EQ(signature.signature, expected_signature);
}

TEST(SignatureTest, CheckQuerySignatureGeneration) {
  sycl::queue queue{sycl::gpu_selector_v};

  auto query_graphs = mbsm::io::loadQueryGraphsFromFile(TEST_QUERY_PATH);

  auto device_query_graph = mbsm::createDeviceQueryGraph(queue, query_graphs);

  mbsm::candidates::Signature* signatures = sycl::malloc_shared<mbsm::candidates::Signature>(device_query_graph.total_nodes, queue);

  auto e = mbsm::isomorphism::filter::generateQuerySignatures(queue, device_query_graph, signatures);

  e.wait_and_throw();

  for (size_t i = 0; i < device_query_graph.total_nodes; ++i) { ASSERT_EQ(signatures[i].signature, expected_query_signatures[i].signature); }

  sycl::free(signatures, queue);
}

TEST(SignatureTest, CheckDataSignatureGeneration) {
  sycl::queue queue{sycl::gpu_selector_v};

  auto data_graphs = mbsm::io::loadDataGraphsFromFile(TEST_DATA_PATH);

  auto device_data_graph = mbsm::createDeviceDataGraph(queue, data_graphs);

  mbsm::candidates::Signature* signatures = sycl::malloc_shared<mbsm::candidates::Signature>(device_data_graph.total_nodes, queue);

  auto e = mbsm::isomorphism::filter::generateDataSignatures(queue, device_data_graph, signatures);

  e.wait_and_throw();

  for (size_t i = 0; i < device_data_graph.total_nodes; ++i) { ASSERT_EQ(signatures[i].signature, expected_data_signatures[i].signature); }

  sycl::free(signatures, queue);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}