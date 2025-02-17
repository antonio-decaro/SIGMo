#include "./include/data.hpp"
#include "./include/utils.hpp"
#include "gtest/gtest.h"
#include <bitset>
#include <mbsm.hpp>
#include <set>
#include <sycl/sycl.hpp>

TEST(FilterTest, SingleFilter) {
  auto query_graphs = mbsm::io::loadQueryGraphsFromFile(TEST_QUERY_PATH);
  auto data_graphs = mbsm::io::loadDataGraphsFromFile(TEST_DATA_PATH);

  sycl::queue queue{sycl::gpu_selector_v};

  auto device_query_graph = mbsm::createDeviceQueryGraph(queue, query_graphs);
  auto device_data_graph = mbsm::createDeviceDataGraph(queue, data_graphs);

  mbsm::candidates::Signature<>* query_signatures = sycl::malloc_shared<mbsm::candidates::Signature<>>(device_query_graph.total_nodes, queue);
  mbsm::candidates::Signature<>* data_signatures = sycl::malloc_shared<mbsm::candidates::Signature<>>(device_data_graph.total_nodes, queue);

  auto e1 = mbsm::isomorphism::filter::generateQuerySignatures(queue, device_query_graph, query_signatures, 0);
  auto e2 = mbsm::isomorphism::filter::generateDataSignatures(queue, device_data_graph, data_signatures, 0);

  queue.wait();

  mbsm::candidates::Candidates candidates{device_query_graph.total_nodes, device_data_graph.total_nodes};
  candidates.candidates = sycl::malloc_shared<mbsm::types::candidates_t>(candidates.getAllocationSize(), queue);
  queue.fill(candidates.candidates, 0, candidates.getAllocationSize());

  auto e3 = mbsm::isomorphism::filter::filterCandidates(queue, device_query_graph, device_data_graph, query_signatures, data_signatures, candidates);
  e3.wait();

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
      for (mbsm::types::label_t l = 0; l < mbsm::candidates::Signature<>::getMaxLabels(); l++) {
        insert &= query_signature.getLabelCount(l) <= data_signature.getLabelCount(l);
        if (!insert) break;
      }
      if (insert) { expected_candidates[query_node].push_back(data_node); }
    }
  }

  for (int i = 0; i < device_query_graph.total_nodes; i++) {
    auto expected = expected_candidates[i];
    for (int j = 0; j < expected.size(); j++) { ASSERT_TRUE(candidates.contains(i, expected[j])); }
  }

  sycl::free(query_signatures, queue);
  sycl::free(data_signatures, queue);
  sycl::free(candidates.candidates, queue);
  mbsm::destroyDeviceDataGraph(device_data_graph, queue);
  mbsm::destroyDeviceQueryGraph(device_query_graph, queue);
}


TEST(FilterTest, RefinementTest) {
  auto query_graphs = mbsm::io::loadQueryGraphsFromFile(TEST_QUERY_PATH);
  auto data_graphs = mbsm::io::loadDataGraphsFromFile(TEST_DATA_PATH);

  sycl::queue queue{sycl::gpu_selector_v};

  auto device_query_graph = mbsm::createDeviceQueryGraph(queue, query_graphs);
  auto device_data_graph = mbsm::createDeviceDataGraph(queue, data_graphs);

  mbsm::candidates::Signature<>* query_signatures = sycl::malloc_shared<mbsm::candidates::Signature<>>(device_query_graph.total_nodes, queue);
  mbsm::candidates::Signature<>* data_signatures = sycl::malloc_shared<mbsm::candidates::Signature<>>(device_data_graph.total_nodes, queue);

  auto e1 = mbsm::isomorphism::filter::generateQuerySignatures(queue, device_query_graph, query_signatures, 0);
  auto e2 = mbsm::isomorphism::filter::generateDataSignatures(queue, device_data_graph, data_signatures, 0);

  queue.wait();

  mbsm::candidates::Candidates candidates{device_query_graph.total_nodes, device_data_graph.total_nodes};
  candidates.candidates = sycl::malloc_shared<mbsm::types::candidates_t>(candidates.getAllocationSize(), queue);
  queue.fill(candidates.candidates, 0, candidates.getAllocationSize());
  queue.wait();

  mbsm::isomorphism::filter::filterCandidates(queue, device_query_graph, device_data_graph, query_signatures, data_signatures, candidates).wait();

  mbsm::isomorphism::filter::refineCandidates(queue, device_query_graph, device_data_graph, query_signatures, data_signatures, candidates).wait();

  // creating a temporary vector to store the candidates
  std::unordered_map<mbsm::types::node_t, std::set<mbsm::types::node_t>> expected_prev_candidates;
  std::unordered_map<mbsm::types::node_t, std::set<mbsm::types::node_t>> expected_curr_candidates;
  for (int i = 0; i < device_query_graph.total_nodes; i++) { expected_prev_candidates[i] = std::set<mbsm::types::node_t>(); }
  for (int i = 0; i < device_query_graph.total_nodes; i++) { expected_curr_candidates[i] = std::set<mbsm::types::node_t>(); }

  for (int data_node = 0; data_node < device_data_graph.total_nodes; data_node++) {
    for (int query_node = 0; query_node < device_query_graph.total_nodes; query_node++) {
      auto query_signature = query_signatures[query_node];
      auto data_signature = data_signatures[data_node];
      auto data_label = device_data_graph.labels[data_node];
      auto query_label = device_query_graph.labels[query_node];

      if (data_label != query_label) { continue; }
      bool insert = true;
      for (mbsm::types::label_t l = 0; l < mbsm::candidates::Signature<>::getMaxLabels(); l++) {
        insert &= query_signature.getLabelCount(l) <= data_signature.getLabelCount(l);
        if (!insert) break;
      }
      if (insert) { expected_prev_candidates[query_node].insert(data_node); }
    }
  }

  for (int data_node = 0; data_node < device_data_graph.total_nodes; data_node++) {
    for (int query_node = 0; query_node < device_query_graph.total_nodes; query_node++) {
      auto query_signature = query_signatures[query_node];
      auto data_signature = data_signatures[data_node];
      auto data_label = device_data_graph.labels[data_node];
      auto query_label = device_query_graph.labels[query_node];

      std::find(expected_prev_candidates[query_node].begin(), expected_prev_candidates[query_node].end(), data_node);
      if (data_label != query_label || expected_prev_candidates[query_node].find(data_node) != expected_prev_candidates[query_node].end()) {
        continue;
      }
      bool insert = true;
      for (mbsm::types::label_t l = 0; l < mbsm::candidates::Signature<>::getMaxLabels(); l++) {
        insert &= query_signature.getLabelCount(l) <= data_signature.getLabelCount(l);
        if (!insert) break;
      }
      if (insert) { expected_curr_candidates[query_node].insert(data_node); }
    }
  }

  for (int i = 0; i < device_query_graph.total_nodes; i++) {
    auto expected = expected_curr_candidates[i];
    for (auto data_node : expected) { ASSERT_TRUE(candidates.contains(i, data_node)); }
  }

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