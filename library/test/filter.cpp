/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./include/data.hpp"
#include "./include/utils.hpp"
#include "gtest/gtest.h"
#include <bitset>
#include <mbsm.hpp>
#include <set>
#include <sycl/sycl.hpp>

TEST(FilterTest, SingleFilter) {
  auto query_graphs = mbsm::io::loadAMGraphsFromFile(TEST_QUERY_PATH);
  auto data_graphs = mbsm::io::loadCSRGraphsFromFile(TEST_DATA_PATH);

  sycl::queue queue{sycl::gpu_selector_v};

  auto device_query_graph = mbsm::createDeviceAMGraph(queue, query_graphs);
  auto device_data_graph = mbsm::createDeviceCSRGraph(queue, data_graphs);

  mbsm::signature::Signature<> signatures{queue, device_data_graph.total_nodes, device_query_graph.total_nodes};

  auto e1 = signatures.generateAMSignatures(device_query_graph);
  auto e2 = signatures.generateCSRSignatures(device_data_graph);

  queue.wait();

  mbsm::candidates::Candidates candidates{queue, device_query_graph.total_nodes, device_data_graph.total_nodes};

  auto e3 = mbsm::isomorphism::filter::filterCandidates(queue, device_query_graph, device_data_graph, signatures, candidates);
  e3.wait();

  auto query_signatures = signatures.getDeviceQuerySignatures();
  auto data_signatures = signatures.getDeviceDataSignatures();

  // creating a temporary vector to store the candidates
  std::unordered_map<mbsm::types::node_t, std::vector<mbsm::types::node_t>> expected_candidates;
  for (int i = 0; i < device_query_graph.total_nodes; i++) { expected_candidates[i] = std::vector<mbsm::types::node_t>(); }

  for (int data_node = 0; data_node < device_data_graph.total_nodes; data_node++) {
    for (int query_node = 0; query_node < device_query_graph.total_nodes; query_node++) {
      auto query_signature = query_signatures[query_node];
      auto data_signature = data_signatures[data_node];
      auto data_label = device_data_graph.labels[data_node];
      auto query_label = device_query_graph.labels[query_node];

      if (data_label != query_label) { continue; }
      bool insert = true;
      for (mbsm::types::label_t l = 0; l < signatures.getMaxLabels(); l++) {
        insert &= query_signature.getLabelCount(l) <= data_signature.getLabelCount(l);
        if (!insert) break;
      }
      if (insert) { expected_candidates[query_node].push_back(data_node); }
    }
  }

  auto device_candidates = candidates.getCandidatesDevice();

  for (int i = 0; i < device_query_graph.total_nodes; i++) {
    auto expected = expected_candidates[i];
    for (int j = 0; j < expected.size(); j++) { ASSERT_TRUE(device_candidates.contains(i, expected[j])); }
  }

  mbsm::destroyDeviceCSRGraph(device_data_graph, queue);
  mbsm::destroyDeviceAMGraph(device_query_graph, queue);
}


TEST(FilterTest, RefinementTest) {
  auto query_graphs = mbsm::io::loadAMGraphsFromFile(TEST_QUERY_PATH);
  auto data_graphs = mbsm::io::loadCSRGraphsFromFile(TEST_DATA_PATH);

  sycl::queue queue{sycl::gpu_selector_v};

  auto device_query_graph = mbsm::createDeviceAMGraph(queue, query_graphs);
  auto device_data_graph = mbsm::createDeviceCSRGraph(queue, data_graphs);

  mbsm::signature::Signature<> signatures{queue, device_data_graph.total_nodes, device_query_graph.total_nodes};

  auto e1 = signatures.generateAMSignatures(device_query_graph);
  auto e2 = signatures.generateCSRSignatures(device_data_graph);


  mbsm::candidates::Candidates candidates{queue, device_query_graph.total_nodes, device_data_graph.total_nodes};

  mbsm::isomorphism::filter::filterCandidates(queue, device_query_graph, device_data_graph, signatures, candidates).wait();

  std::unordered_map<mbsm::types::node_t, std::set<mbsm::types::node_t>> expected_candidates;
  for (int i = 0; i < device_query_graph.total_nodes; i++) { expected_candidates[i] = std::set<mbsm::types::node_t>(); }

  auto query_signatures = signatures.getDeviceQuerySignatures();
  auto data_signatures = signatures.getDeviceDataSignatures();

  for (int data_node = 0; data_node < device_data_graph.total_nodes; data_node++) {
    for (int query_node = 0; query_node < device_query_graph.total_nodes; query_node++) {
      auto query_signature = query_signatures[query_node];
      auto data_signature = data_signatures[data_node];
      auto data_label = device_data_graph.labels[data_node];
      auto query_label = device_query_graph.labels[query_node];

      if (data_label != query_label) { continue; }
      bool insert = true;
      for (mbsm::types::label_t l = 0; l < signatures.getMaxLabels(); l++) {
        insert = insert && (query_signature.getLabelCount(l) <= data_signature.getLabelCount(l));
        if (!insert) break;
      }
      if (insert) { expected_candidates[query_node].insert(data_node); }
    }
  }


  signatures.refineCSRSignatures(device_data_graph).wait();
  signatures.refineAMSignatures(device_query_graph).wait();

  mbsm::isomorphism::filter::refineCandidates(queue, device_query_graph, device_data_graph, signatures, candidates).wait();

  auto expected_query_signatures = getExpectedQuerySignatures(TEST_QUERY_PATH, 1);
  auto expected_data_signatures = getExpectedDataSignatures(TEST_DATA_PATH, 1);
  for (int data_node = 0; data_node < device_data_graph.total_nodes; data_node++) {
    for (int query_node = 0; query_node < device_query_graph.total_nodes; query_node++) {
      auto query_signature = expected_query_signatures[query_node];
      auto data_signature = expected_data_signatures[data_node];
      auto data_label = device_data_graph.labels[data_node];
      auto query_label = device_query_graph.labels[query_node];

      if (expected_candidates[query_node].find(data_node) == expected_candidates[query_node].end()) { continue; }

      bool keep = query_label == data_label;
      for (mbsm::types::label_t l = 0; l < signatures.getMaxLabels() && keep; l++) {
        keep = keep && (query_signature.getLabelCount(l) <= data_signature.getLabelCount(l));
      }
      if (!keep) { expected_candidates[query_node].erase(data_node); }
    }
  }

  auto candidates_device = candidates.getCandidatesDevice();

  for (int i = 0; i < device_query_graph.total_nodes; i++) {
    ASSERT_EQ(expected_candidates[i].size(), candidates.getCandidatesCount(i));
    auto expected = expected_candidates[i];
    for (auto data_node : expected) { ASSERT_TRUE(candidates_device.contains(i, data_node)); }
  }

  mbsm::destroyDeviceCSRGraph(device_data_graph, queue);
  mbsm::destroyDeviceAMGraph(device_query_graph, queue);
}


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}