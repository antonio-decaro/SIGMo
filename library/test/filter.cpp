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
  mbsm::candidates::Candidates expected_candidates{device_query_graph.total_nodes, device_data_graph.total_nodes};
  expected_candidates.candidates = new mbsm::types::candidates_t[expected_candidates.getAllocationSize()];
  for (size_t i = 0; i < expected_candidates.getAllocationSize(); i++) { expected_candidates.candidates[i] = 0; }

  for (int data_node = 0; data_node < device_data_graph.total_nodes; data_node++) {
    for (int query_node = 0; query_node < device_query_graph.total_nodes; query_node++) {
      auto query_signature = query_signatures[query_node];
      auto data_signature = data_signatures[data_node];

      bool insert = true;
      for (mbsm::types::label_t l = 0; l < mbsm::candidates::Signature<>::getMaxLabels(); l++) {
        insert &= query_signature.getLabelCount(l) <= data_signature.getLabelCount(l);
        if (!insert) break;
      }
      if (insert) { expected_candidates.insert(query_node, data_node); }
    }
  }

  for (size_t i = 0; i < candidates.getAllocationSize(); i++) { ASSERT_EQ(candidates.candidates[i], expected_candidates.candidates[i]); }

  // TODO add assertions

  sycl::free(query_signatures, queue);
  sycl::free(data_signatures, queue);
  sycl::free(candidates.candidates, queue);
  mbsm::destroyDeviceDataGraph(device_data_graph, queue);
  mbsm::destroyDeviceQueryGraph(device_query_graph, queue);
  delete expected_candidates.candidates;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}