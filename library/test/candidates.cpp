/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./include/data.hpp"
#include "./include/utils.hpp"
#include "gtest/gtest.h"
#include <bitset>
#include <sigmo.hpp>
#include <sycl/sycl.hpp>

TEST(SignatureTest, CheckSignatureMethods) {
  sigmo::signature::Signature<>::SignatureDevice signature;

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

  auto query_graphs = sigmo::io::loadAMGraphsFromFile(TEST_QUERY_PATH);

  auto device_query_graph = sigmo::createDeviceAMGraph(queue, query_graphs);

  sigmo::signature::Signature<> signatures{queue, device_query_graph.total_nodes, device_query_graph.total_nodes};

  auto e = signatures.generateAMSignatures(device_query_graph);

  auto device_signatures = signatures.getDeviceQuerySignatures();

  e.wait();

  auto expected_query_signatures = getExpectedQuerySignatures(TEST_QUERY_PATH, 0);

  for (size_t i = 0; i < device_query_graph.total_nodes; ++i) { ASSERT_EQ(device_signatures[i].signature, expected_query_signatures[i].signature); }
}

TEST(SignatureTest, RefineQuerySignature) {
  sycl::queue queue{sycl::gpu_selector_v};

  auto query_graphs = sigmo::io::loadAMGraphsFromFile(TEST_QUERY_PATH);

  auto device_query_graph = sigmo::createDeviceAMGraph(queue, query_graphs);

  sigmo::signature::Signature<> signatures{queue, device_query_graph.total_nodes, device_query_graph.total_nodes};

  auto e = signatures.generateAMSignatures(device_query_graph);
  e.wait();

  auto device_signatures = signatures.getDeviceQuerySignatures();

  for (int i = 0; i < 10; i++) {
    e = signatures.refineAMSignatures(device_query_graph);
    e.wait();
    auto expected_query_signatures = getExpectedQuerySignatures(TEST_QUERY_PATH, i + 1);
    for (size_t i = 0; i < device_query_graph.total_nodes; ++i) { ASSERT_EQ(device_signatures[i].signature, expected_query_signatures[i].signature); }
  }
}


TEST(SignatureTest, CheckDataSignatureGeneration) {
  sycl::queue queue{sycl::gpu_selector_v};

  auto data_graphs = sigmo::io::loadCSRGraphsFromFile(TEST_DATA_PATH);

  auto device_data_graph = sigmo::createDeviceCSRGraph(queue, data_graphs);

  sigmo::signature::Signature<> signatures{queue, device_data_graph.total_nodes, device_data_graph.total_nodes};

  auto e = signatures.generateCSRSignatures(device_data_graph);

  e.wait();

  auto device_signatures = signatures.getDeviceDataSignatures();

  auto expected_data_signatures = getExpectedDataSignatures(TEST_DATA_PATH, 0);

  for (size_t i = 0; i < device_data_graph.total_nodes; ++i) { ASSERT_EQ(device_signatures[i].signature, expected_data_signatures[i].signature); }
}

TEST(SignatureTest, RefineDataSignature) {
  sycl::queue queue{sycl::gpu_selector_v};

  auto data_graphs = sigmo::io::loadCSRGraphsFromFile(TEST_DATA_PATH);

  auto device_data_graph = sigmo::createDeviceCSRGraph(queue, data_graphs);

  sigmo::signature::Signature<> signatures{queue, device_data_graph.total_nodes, device_data_graph.total_nodes};

  auto e = signatures.generateCSRSignatures(device_data_graph);
  e.wait();

  auto device_signatures = signatures.getDeviceDataSignatures();

  for (int i = 0; i < 10; i++) {
    e = signatures.refineCSRSignatures(device_data_graph);
    e.wait();
    auto expected_data_signatures = getExpectedDataSignatures(TEST_DATA_PATH, i + 1);
    for (size_t i = 0; i < device_data_graph.total_nodes; ++i) { ASSERT_EQ(device_signatures[i].signature, expected_data_signatures[i].signature); }
  }
}

TEST(CandidateTest, CheckInsertAndRemove) {
  sycl::queue queue{sycl::gpu_selector_v};
  const size_t num_nodes = 128;
  sigmo::candidates::Candidates candidates{queue, 2, num_nodes};

  auto device_candidates = candidates.getCandidatesDevice();

  device_candidates.insert(0, 0);
  device_candidates.insert(0, 31);
  device_candidates.insert(0, 32);
  device_candidates.insert(0, 124);

  ASSERT_EQ(device_candidates.candidates[0], 0b0000000000000000000000000000000110000000000000000000000000000001u);
  ASSERT_EQ(device_candidates.candidates[1], 0b0001000000000000000000000000000000000000000000000000000000000000u);
  ASSERT_EQ(device_candidates.getCandidatesCount(0), 4);

  device_candidates.remove(0, 32);
  ASSERT_EQ(device_candidates.candidates[0], 0b0000000000000000000000000000000010000000000000000000000000000001u);

  ASSERT_EQ(device_candidates.getCandidatesCount(0), 3);

  device_candidates.insert(1, 0);
  ASSERT_EQ(device_candidates.candidates[2], 0b0000000000000000000000000000000000000000000000000000000000000001u);
  ASSERT_EQ(device_candidates.candidates[3], 0b0u);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}