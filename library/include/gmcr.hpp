#pragma once

#include "candidates.hpp"
#include "device.hpp"
#include "graph.hpp"
#include "pool.hpp"
#include "signature.hpp"
#include "types.hpp"
#include "utils.hpp"
#include <sycl/sycl.hpp>

namespace sigmo {
namespace isomorphism {
namespace mapping {

class GMCR {
private:
  struct GMCRDevice {
    uint32_t* data_graph_offsets;
    uint32_t* query_graph_indices;
  } gmcr;
  sycl::queue& queue;

public:
  GMCR(sycl::queue& queue) : queue(queue) {}
  ~GMCR() {
    sycl::free(gmcr.data_graph_offsets, queue);
    sycl::free(gmcr.query_graph_indices, queue);
  }

  // Offloaded version of generateGMCR using SYCL kernels.
  utils::BatchedEvent
  generateGMCR(sigmo::DeviceBatchedCSRGraph& query_graphs, sigmo::DeviceBatchedCSRGraph& data_graphs, sigmo::candidates::Candidates& candidates) {
    // Get dimensions
    const size_t total_query_graphs = query_graphs.num_graphs;
    const size_t total_data_graphs = data_graphs.num_graphs;

    // Allocate device memory for data_graph_offsets (size = total_data_graphs+1)
    uint32_t* d_data_graph_offsets = sycl::malloc_shared<uint32_t>(total_data_graphs + 1, queue);
    // Initialize to zero
    queue.fill(d_data_graph_offsets, 0, total_data_graphs + 1).wait();

    // --- Kernel 1: Count query graphs (with > 1 node) per data graph ---
    // For each pair (query_graph, data_graph), if every node in the query graph
    // has a candidate in the given data graph, then atomically increment the counter.
    auto k1 = queue.parallel_for(
        sycl::range<2>(total_query_graphs, total_data_graphs),
        [=, query_graphs = query_graphs, data_graphs = data_graphs, candidates = candidates.getCandidatesDevice()](sycl::id<2> idx) {
          size_t query_graph_id = idx[0];
          size_t data_graph_id = idx[1];
          size_t start_data = data_graphs.graph_offsets[data_graph_id];
          size_t end_data = data_graphs.graph_offsets[data_graph_id + 1];
          size_t num_query_nodes = query_graphs.getGraphNodes(query_graph_id);

          size_t offset_query_nodes = query_graphs.getPreviousNodes(query_graph_id);
          bool add = true;
          for (size_t i = 0; i < num_query_nodes && add; i++) {
            size_t global_query_node = offset_query_nodes + i;
            add = add && (candidates.getCandidatesCount(global_query_node, start_data, end_data) > 0);
          }
          if (add) {
            // Increment the counter for this data graph.
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_offset(
                d_data_graph_offsets[data_graph_id + 1]);
            atomic_offset.fetch_add(1);
          }
        });
    k1.wait();

    // --- Kernel 2: Compute prefix sum of data_graph_offsets ---
    // (Here we assume total_data_graphs is small so a single task is acceptable.)
    auto k2 = queue.single_task([=]() {
      for (size_t i = 1; i <= total_data_graphs; i++) { d_data_graph_offsets[i] += d_data_graph_offsets[i - 1]; }
    });
    k2.wait();

    // The total number of query indices is now the last element in d_data_graph_offsets.
    uint32_t total_query_indices;
    queue.copy(&d_data_graph_offsets[total_data_graphs], &total_query_indices, 1).wait();

    // Allocate device memory for query_graph_indices.
    uint32_t* d_query_graph_indices = sycl::malloc_shared<uint32_t>(total_query_indices, queue);

    // Create a copy of the prefix sum array to serve as atomic “current offsets”
    uint32_t* current_offsets = sycl::malloc_shared<uint32_t>(total_data_graphs + 1, queue);
    queue.copy(d_data_graph_offsets, current_offsets, total_data_graphs + 1).wait();

    // --- Kernel 3: Fill query_graph_indices ---
    // For each (query_graph, data_graph) pair (with query graphs having > 1 node),
    // if the graph qualifies (each query node has a candidate in the data graph),
    // then atomically obtain an index (from current_offsets) and write the query_graph_id.
    auto k3 = queue.parallel_for(
        sycl::range<2>(total_query_graphs, total_data_graphs),
        [=, query_graphs = query_graphs, data_graphs = data_graphs, candidates = candidates.getCandidatesDevice()](sycl::id<2> idx) {
          size_t query_graph_id = idx[0];
          size_t data_graph_id = idx[1];
          size_t num_query_nodes = query_graphs.getGraphNodes(query_graph_id);
          if (num_query_nodes <= 1) return; // skip
          size_t offset_query_nodes = query_graphs.getPreviousNodes(query_graph_id);
          bool add = true;
          for (size_t i = 0; i < num_query_nodes && add; i++) {
            size_t global_query_node = offset_query_nodes + i;
            size_t start_data = data_graphs.graph_offsets[data_graph_id];
            size_t end_data = data_graphs.graph_offsets[data_graph_id + 1];
            add = add && (candidates.getCandidatesCount(global_query_node, start_data, end_data) > 0);
          }
          if (add) {
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_off(
                current_offsets[data_graph_id]);
            uint32_t index = atomic_off.fetch_add(1);
            d_query_graph_indices[index] = static_cast<uint32_t>(query_graph_id);
          }
        });
    k3.wait();

    // Build the result structure.
    gmcr.data_graph_offsets = d_data_graph_offsets;
    gmcr.query_graph_indices = d_query_graph_indices;
    sycl::free(current_offsets, queue);

    utils::BatchedEvent ret;
    ret.add(k1);
    ret.add(k2);
    ret.add(k3);
    return ret;
  }

  GMCRDevice getGMCRDevice() { return gmcr; }
};


} // namespace mapping
} // namespace isomorphism
} // namespace sigmo