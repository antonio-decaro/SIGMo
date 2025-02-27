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

template<candidates::CandidatesDomain D = candidates::CandidatesDomain::Query>
utils::BatchedEvent filterCandidates(sycl::queue& queue,
                                     mbsm::DeviceBatchedQueryGraph& query_graph,
                                     mbsm::DeviceBatchedDataGraph& data_graph,
                                     mbsm::signature::Signature<>* query_signatures,
                                     mbsm::signature::Signature<>* data_signatures,
                                     mbsm::candidates::Candidates candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;
  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<mbsm::device::kernels::FilterCandidatesKernel<D>>(sycl::range<1>(total_data_nodes), [=](sycl::item<1> item) {
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
        if (insert) {
          if constexpr (D == candidates::CandidatesDomain::Data) {
            candidates.insert(data_node_id, query_node_id);
          } else {
            candidates.atomicInsert(query_node_id, data_node_id);
          }
        }
      }
    });
  });

  utils::BatchedEvent be;
  be.add(e);
  return be;
}

template<candidates::CandidatesDomain D = candidates::CandidatesDomain::Query>
utils::BatchedEvent refineCandidates(sycl::queue& queue,
                                     mbsm::DeviceBatchedQueryGraph& query_graph,
                                     mbsm::DeviceBatchedDataGraph& data_graph,
                                     mbsm::signature::Signature<>* query_signatures,
                                     mbsm::signature::Signature<>* data_signatures,
                                     mbsm::candidates::Candidates candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;
  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<mbsm::device::kernels::RefineCandidatesKernel<D>>(sycl::range<1>(total_data_nodes), [=](sycl::item<1> item) {
      auto data_node_id = item.get_id(0);
      auto data_signature = data_signatures[data_node_id];

      for (size_t query_node_id = 0; query_node_id < total_query_nodes; ++query_node_id) {
        if constexpr (D == candidates::CandidatesDomain::Data) {
          if (!candidates.contains(data_node_id, query_node_id)) { continue; }
        } else {
          if (!candidates.atomicContains(query_node_id, data_node_id)) { continue; }
        }
        auto query_signature = query_signatures[query_node_id];

        bool keep = true;
        for (types::label_t l = 0; l < signature::Signature<>::getMaxLabels() && keep; l++) {
          keep = keep && (query_signature.getLabelCount(l) <= data_signature.getLabelCount(l));
        }
        if (!keep) {
          if constexpr (D == candidates::CandidatesDomain::Data) {
            candidates.remove(data_node_id, query_node_id);
          } else {
            candidates.atomicRemove(query_node_id, data_node_id);
          }
        }
      }
    });
  });

  utils::BatchedEvent be;
  be.add(e);
  return be;
}

} // namespace filter

namespace join {

utils::BatchedEvent joinCandidates(sycl::queue& queue,
                                   mbsm::DeviceBatchedQueryGraph& query_graph,
                                   mbsm::DeviceBatchedDataGraph& data_graph,
                                   mbsm::candidates::Candidates candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;
  size_t total_query_graphs = query_graph.num_graphs;
  size_t total_data_graphs = data_graph.num_graphs;
}

} // namespace join

} // namespace isomorphism
} // namespace mbsm