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

  std::cout << "Reading from " << TEST_QUERY_PATH << std::endl;

  auto query_graphs = mbsm::io::loadQueryGraphsFromFile(TEST_QUERY_PATH);
  std::cout << "Loaded " << query_graphs.size() << " query graphs" << std::endl;

  auto device_query_graph = mbsm::createDeviceQueryGraph(queue, query_graphs);

  std::cout << "Created device query graph" << std::endl;

  mbsm::candidates::Signature* signatures = sycl::malloc_shared<mbsm::candidates::Signature>(device_query_graph.total_nodes, queue);

  std::cout << "Allocated memory for signatures" << std::endl;

  auto e = mbsm::isomorphism::filter::generateQuerySignatures(queue, device_query_graph, signatures);

  std::cout << "Generated query signatures" << std::endl;

  e.wait_and_throw();

  std::cout << "Qeueue terminated" << std::endl;


  mbsm::candidates::Signature expected_signatures[]{// Graph 1
                                                    {0b00000000000000000000001100000000},
                                                    {0b00000000000000000000001000000000},
                                                    {0b00000000000000000000001000000000},
                                                    {0b00000000000000000000001100000000},
                                                    {0b00000000000000000000001000000100},
                                                    {0b00000000000000000000000100000100},
                                                    // Graph 2
                                                    {0b00000000000000010010000000001000},
                                                    {0b00000000000000010001000000000000},
                                                    {0b00000000000000010001000000000000},
                                                    {0b00000000000000010000000000000100},
                                                    {0b00000000000000010000000000000100},
                                                    // Graph 3
                                                    {0b00000000000000000000010000000100},
                                                    {0b00000000000000000000010000001000},
                                                    {0b00000000000000000000000000001100},
                                                    {0b00000000000000000000000000001100},
                                                    {0b00000000000000000000000000001100},
                                                    {0b00000000000000000000000000001100},
                                                    {0b00000000000000000000000000001100},
                                                    {0b00000000000000000000000000001100},
                                                    {0b00000000000000000001000000001000},
                                                    {0b00000000000000000001000000000100}};

  for (size_t i = 0; i < device_query_graph.total_nodes; ++i) { ASSERT_EQ(signatures[i].signature, expected_signatures[i].signature); }

  std::cout << "Signatures are correct" << std::endl;

  sycl::free(signatures, queue);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}