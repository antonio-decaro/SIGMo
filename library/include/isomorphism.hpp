#pragma once

#include "candidates.hpp"
#include "device.hpp"
#include "graph.hpp"
#include "pool.hpp"
#include "signature.hpp"
#include "types.hpp"
#include <sycl/sycl.hpp>


namespace mbsm {
namespace isomorphism {
namespace filter {

utils::BatchedEvent filterCandidates(sycl::queue& queue,
                                     mbsm::DeviceBatchedQueryGraph& query_graph,
                                     mbsm::DeviceBatchedDataGraph& data_graph,
                                     mbsm::signature::Signature<>* query_signatures,
                                     mbsm::signature::Signature<>* data_signatures,
                                     mbsm::candidates::Candidates candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;
  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<mbsm::device::kernels::FilterCandidatesKernel>(sycl::range<1>(total_data_nodes), [=](sycl::item<1> item) {
      auto data_node_id = item.get_id(0);
      auto data_signature = data_signatures[data_node_id];
      auto query_labels = query_graph.labels;
      auto data_labels = data_graph.labels;

      for (size_t query_node_id = 0; query_node_id < total_query_nodes; ++query_node_id) {
        if (query_labels[query_node_id] != data_labels[data_node_id]) { continue; }
        auto query_signature = query_signatures[query_node_id];

        bool insert = true;
        for (types::label_t l = 0; l < signature::Signature<>::getMaxLabels(); l++) {
          insert = insert && (query_signature.getLabelCount(l) <= data_signature.getLabelCount(l));
          if (!insert) break;
        }
        if (insert) { candidates.atomicInsert(query_node_id, data_node_id); }
      }
    });
  });

  utils::BatchedEvent be;
  be.add(e);
  return be;
}

utils::BatchedEvent refineCandidates(sycl::queue& queue,
                                     mbsm::DeviceBatchedQueryGraph& query_graph,
                                     mbsm::DeviceBatchedDataGraph& data_graph,
                                     mbsm::signature::Signature<>* query_signatures,
                                     mbsm::signature::Signature<>* data_signatures,
                                     mbsm::candidates::Candidates candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;
  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<mbsm::device::kernels::RefineCandidatesKernel>(sycl::range<1>(total_data_nodes), [=](sycl::item<1> item) {
      auto data_node_id = item.get_id(0);
      auto data_signature = data_signatures[data_node_id];
      auto query_labels = query_graph.labels;
      auto data_labels = data_graph.labels;

      for (size_t query_node_id = 0; query_node_id < total_query_nodes; ++query_node_id) {
        if (!candidates.atomicContains(query_node_id, data_node_id)) { continue; }
        auto query_signature = query_signatures[query_node_id];

        bool keep = true;
        for (types::label_t l = 0; l < signature::Signature<>::getMaxLabels() && keep; l++) {
          keep = keep && (query_signature.getLabelCount(l) <= data_signature.getLabelCount(l));
        }
        if (!keep) { candidates.atomicRemove(query_node_id, data_node_id); }
      }
    });
  });

  utils::BatchedEvent be;
  be.add(e);
  return be;
}

} // namespace filter

namespace join {} // namespace join

} // namespace isomorphism
} // namespace mbsm