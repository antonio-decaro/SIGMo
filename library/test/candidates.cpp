#include "./include/data.hpp"
#include "./include/utils.hpp"
#include "gtest/gtest.h"
#include <bitset>
#include <mbsm.hpp>
#include <sycl/sycl.hpp>

TEST(SignatureTest, CheckSignatureMethods) {
  mbsm::candidates::Signature<> signature;

  signature.setLabelCount(0, 1);
  signature.setLabelCount(2, 3);
  signature.setLabelCount(5, 2);
  signature.setLabelCount(15, 1);

  ASSERT_EQ(signature.signature, 0b0001000000000000000000000000000000000000001000000000001100000001);
  ASSERT_EQ(signature.getLabelCount(15), static_cast<uint8_t>(1));

  signature.incrementLabelCount(15);
  ASSERT_EQ(signature.signature, 0b0010000000000000000000000000000000000000001000000000001100000001);

  signature.incrementLabelCount(14);
  ASSERT_EQ(signature.signature, 0b0010000100000000000000000000000000000000001000000000001100000001);

  signature.incrementLabelCount(14, 14);
  ASSERT_EQ(signature.signature, 0b0010111100000000000000000000000000000000001000000000001100000001);
}

TEST(SignatureTest, CheckQuerySignatureGeneration) {
  sycl::queue queue{sycl::gpu_selector_v};

  auto query_graphs = mbsm::io::loadQueryGraphsFromFile(TEST_QUERY_PATH);

  auto device_query_graph = mbsm::createDeviceQueryGraph(queue, query_graphs);

  mbsm::candidates::Signature<>* signatures = sycl::malloc_shared<mbsm::candidates::Signature<>>(device_query_graph.total_nodes, queue);

  auto e = mbsm::isomorphism::filter::generateQuerySignatures(queue, device_query_graph, signatures);

  e.wait();

  auto expected_query_signatures = getExpectedQuerySignatures(TEST_QUERY_PATH);

  for (size_t i = 0; i < device_query_graph.total_nodes; ++i) { ASSERT_EQ(signatures[i].signature, expected_query_signatures[i].signature); }

  sycl::free(signatures, queue);
}

TEST(SignatureTest, CheckDataSignatureGeneration) {
  sycl::queue queue{sycl::gpu_selector_v};

  auto data_graphs = mbsm::io::loadDataGraphsFromFile(TEST_DATA_PATH);

  auto device_data_graph = mbsm::createDeviceDataGraph(queue, data_graphs);

  mbsm::candidates::Signature<>* signatures = sycl::malloc_shared<mbsm::candidates::Signature<>>(device_data_graph.total_nodes, queue);

  auto e = mbsm::isomorphism::filter::generateDataSignatures(queue, device_data_graph, signatures);

  e.wait();

  auto expected_data_signatures = getExpectedDataSignatures(TEST_DATA_PATH);

  for (size_t i = 0; i < device_data_graph.total_nodes; ++i) { ASSERT_EQ(signatures[i].signature, expected_data_signatures[i].signature); }

  sycl::free(signatures, queue);
}

TEST(CandidateTest, CheckInsertAndRemove) {
  sycl::queue queue{sycl::gpu_selector_v};
  const size_t num_nodes = 128;
  mbsm::candidates::Candidates candidates(2, num_nodes);
  candidates.setDataCandidates(sycl::malloc_shared<mbsm::types::candidates_t>(candidates.getAllocationSize(), queue));
  queue.fill(candidates.candidates, static_cast<mbsm::types::candidates_t>(0), candidates.getAllocationSize()).wait();
  candidates.insert(0, 0);
  candidates.insert(0, 31);
  candidates.insert(0, 32);
  candidates.insert(0, 124);

  ASSERT_EQ(candidates.candidates[0], 0b0000000000000000000000000000000110000000000000000000000000000001u);
  ASSERT_EQ(candidates.candidates[1], 0b0001000000000000000000000000000000000000000000000000000000000000u);
  ASSERT_EQ(candidates.getCandidatesCount(0, num_nodes), 4);

  candidates.remove(0, 32);
  ASSERT_EQ(candidates.candidates[0], 0b0000000000000000000000000000000010000000000000000000000000000001u);

  ASSERT_EQ(candidates.getCandidatesCount(0, num_nodes), 3);

  candidates.insert(1, 0);
  ASSERT_EQ(candidates.candidates[2], 0b0000000000000000000000000000000000000000000000000000000000000001u);
  ASSERT_EQ(candidates.candidates[3], 0b0u);

  sycl::free(candidates.candidates, queue);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}