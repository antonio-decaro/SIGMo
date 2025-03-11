/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "candidates.hpp"
#include "device.hpp"
#include "types.hpp"
#include "utils.hpp"
#include <cstdint>
#include <sycl/sycl.hpp>

namespace mbsm {
namespace signature {

enum class Algorithm { ViewBased, PowerGraph };

template<Algorithm A = Algorithm::PowerGraph, size_t Bits = 4>
class Signature {
public:
  struct SignatureDevice {
    uint64_t signature;

    SignatureDevice() : signature(0) {}

    SignatureDevice(uint64_t signature) : signature(signature) {}

    SYCL_EXTERNAL static uint16_t getMaxLabels() { return sizeof(signature) * 8 / Bits; }

    SYCL_EXTERNAL void setLabelCount(uint8_t label, uint8_t count) {
      if (label < (64 / Bits) && count < (1 << Bits)) {
        signature &= ~((static_cast<uint64_t>((1 << Bits) - 1)) << (label * Bits)); // Clear the bits for the label
        signature |= (static_cast<uint64_t>(count) << (label * Bits));              // Set the new count
      }
    }

    SYCL_EXTERNAL uint8_t getLabelCount(uint8_t label) const {
      if (label < (64 / Bits)) { return (signature >> (label * Bits)) & ((1 << Bits) - 1); }
      return 0;
    }

    SYCL_EXTERNAL void incrementLabelCount(uint8_t label, uint8_t add = 1) {
      if (label < (64 / Bits)) {
        uint8_t count = getLabelCount(label);
        if (count < ((1 << Bits) - 1)) { // Ensure count does not exceed max value
          setLabelCount(label, count + static_cast<uint8_t>(add));
        }
      }
    }

    SYCL_EXTERNAL void clear() { signature = 0; }
  };


  // TODO consider to use the shared memory to store the graph to avoid uncoallesced memory access
  utils::BatchedEvent generateQuerySignatures(DeviceBatchedQueryGraph& graphs) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);
    auto e = queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<mbsm::device::kernels::GenerateQuerySignaturesKernel>(
          sycl::range<1>{graphs.total_nodes}, [=, graphs = graphs, signatures = this->query_signatures](sycl::item<1> item) {
            auto node_id = item.get_id(0);
            // Get the neighbors of the current node
            types::node_t neighbors[types::MAX_NEIGHBORS];
            graphs.getNeighbors(node_id, neighbors);
            for (types::node_t i = 0; neighbors[i] != types::NULL_NODE && i < types::MAX_NEIGHBORS; ++i) {
              auto neighbor = neighbors[i];
              signatures[node_id].incrementLabelCount(graphs.labels[neighbor]);
            }
          });
    });
    event.add(e);

    return event;
  }

  template<Algorithm _A = A>
  utils::BatchedEvent refineQuerySignatures(DeviceBatchedQueryGraph& graphs, size_t iter = 1);

  utils::BatchedEvent generateDataSignatures(DeviceBatchedDataGraph& graphs) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);
    sycl::buffer<mbsm::signature::Signature<>, 1> buffer(sycl::range{global_range});
    auto e = queue.submit([&](sycl::handler& cgh) {
      auto* row_offsets = graphs.row_offsets;
      auto* column_indices = graphs.column_indices;
      auto* labels = graphs.labels;

      cgh.parallel_for<mbsm::device::kernels::GenerateDataSignaturesKernel>(global_range,
                                                                            [=, signatures = this->data_signatures](sycl::item<1> item) {
                                                                              auto node_id = item.get_id(0);

                                                                              uint32_t start_neighbor = row_offsets[node_id];
                                                                              uint32_t end_neighbor = row_offsets[node_id + 1];
                                                                              mbsm::types::label_t node_label = labels[node_id];

                                                                              for (uint32_t i = start_neighbor; i < end_neighbor; ++i) {
                                                                                auto neighbor = column_indices[i];
                                                                                signatures[node_id].incrementLabelCount(labels[neighbor]);
                                                                              }
                                                                            });
    });
    event.add(e);

    return event;
  }

  template<Algorithm _A = A>
  utils::BatchedEvent refineDataSignatures(DeviceBatchedDataGraph& graphs, size_t iter = 1);

  Signature(sycl::queue& queue, size_t data_nodes, size_t query_nodes) : queue(queue), data_nodes(data_nodes), query_nodes(query_nodes) {
    data_signatures = sycl::malloc_shared<SignatureDevice>(data_nodes, queue);
    query_signatures = sycl::malloc_shared<SignatureDevice>(query_nodes, queue);
    tmp_buff = sycl::malloc_shared<SignatureDevice>(std::max(data_nodes, query_nodes), queue);
    queue.fill(data_signatures, 0, data_nodes).wait();
    queue.fill(query_signatures, 0, query_nodes).wait();
  }

  ~Signature() {
    sycl::free(data_signatures, queue);
    sycl::free(query_signatures, queue);
    sycl::free(tmp_buff, queue);
  }

  size_t getDataSignatureAllocationSize() const { return data_nodes * sizeof(SignatureDevice); }
  size_t getQuerySignatureAllocationSize() const { return query_nodes * sizeof(SignatureDevice); }
  SignatureDevice* getDeviceDataSignatures() const { return data_signatures; }
  SignatureDevice* getDeviceQuerySignatures() const { return query_signatures; }
  size_t getMaxLabels() const { return SignatureDevice::getMaxLabels(); }

private:
  sycl::queue& queue;
  size_t data_nodes;
  size_t query_nodes;
  SignatureDevice* data_signatures;
  SignatureDevice* query_signatures;
  SignatureDevice* tmp_buff;

  template<>
  utils::BatchedEvent refineQuerySignatures<Algorithm::ViewBased>(DeviceBatchedQueryGraph& graphs, size_t iter) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);

    auto copy_event = queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>(graphs.total_nodes), [=, tmp_buff = this->tmp_buff, signatures = this->query_signatures](sycl::item<1> item) {
        tmp_buff[item] = signatures[item];
      });
    });
    event.add(copy_event);
    copy_event.wait();

    auto refinement_event = queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(copy_event);
      const uint16_t max_labels_count = Signature::SignatureDevice::getMaxLabels();

      cgh.parallel_for<mbsm::device::kernels::RefineQuerySignaturesKernel>(
          sycl::range<1>{graphs.total_nodes},
          [=, graphs = graphs, signatures = this->query_signatures, tmp_buff = this->tmp_buff](sycl::item<1> item) {
            auto node_id = item.get_id(0);
            // Get the neighbors of the current node
            types::node_t neighbors[types::MAX_NEIGHBORS];
            types::label_t node_label = graphs.labels[node_id];
            graphs.getNeighbors(node_id, neighbors);
            for (types::node_t i = 0; neighbors[i] != types::NULL_NODE && i < types::MAX_NEIGHBORS; ++i) {
              auto neighbor = neighbors[i];
              for (types::label_t l = 0; l < max_labels_count; l++) {
                auto count = tmp_buff[neighbor].getLabelCount(l);
                if (l == node_label) { count -= iter; }
                if (count > 0) signatures[node_id].incrementLabelCount(l, count);
              }
            }
          });
    });
    event.add(refinement_event);
    return event;
  }

  template<>
  utils::BatchedEvent refineQuerySignatures<Algorithm::PowerGraph>(DeviceBatchedQueryGraph& graphs, size_t iter) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);
    const uint16_t max_labels_count = Signature::SignatureDevice::getMaxLabels();

    auto refinement_event = queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<mbsm::device::kernels::RefineQuerySignaturesKernel>(
          sycl::range<1>{graphs.total_nodes},
          [=, graphs = graphs, signatures = this->query_signatures, tmp_buff = this->tmp_buff](sycl::item<1> item) {
            auto node_id = item.get_id(0);
            auto graph_id = graphs.getGraphId(node_id);
            auto prev_nodes = graphs.getPreviousNodes(graph_id);

            types::node_t neighbors[types::MAX_NEIGHBORS];
            utils::detail::Bitset<uint32_t> frontier, reachable;

            frontier.set(node_id - prev_nodes);
            reachable.set(node_id - prev_nodes);
            for (uint curr_iter = 0; curr_iter < iter && !frontier.empty(); curr_iter++) {
              utils::detail::Bitset<uint32_t> next_frontier;

              for (uint idx = 0; idx < frontier.size(); idx++) {
                auto u = frontier.getSetBit(idx);
                graphs.getNeighbors(u + prev_nodes, neighbors, graph_id, prev_nodes);
                for (uint i = 0; neighbors[i] != types::NULL_NODE && i < types::MAX_NEIGHBORS; ++i) {
                  auto v = neighbors[i] - prev_nodes;
                  if (!reachable.get(v)) {
                    reachable.set(v);
                    next_frontier.set(v);
                  }
                }
              }
              frontier = next_frontier;
            }
            reachable.unset(node_id - prev_nodes);
            signatures[node_id].clear();
            for (uint idx = 0; idx < reachable.size(); idx++) {
              auto u = reachable.getSetBit(idx) + prev_nodes;
              types::label_t u_label = graphs.labels[u];
              signatures[node_id].incrementLabelCount(u_label);
            }
          });
    });
    event.add(refinement_event);
    return event;
  }

  template<>
  utils::BatchedEvent refineDataSignatures<Algorithm::ViewBased>(DeviceBatchedDataGraph& graphs, size_t view_size) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);
    auto signatures = this->data_signatures;
    auto tmp_buff = this->tmp_buff;

    auto copy_event = queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>(graphs.total_nodes), [=](sycl::item<1> item) { tmp_buff[item] = signatures[item]; });
    });
    event.add(copy_event);
    auto refine_event = queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(copy_event);
      auto* row_offsets = graphs.row_offsets;
      auto* column_indices = graphs.column_indices;
      auto* labels = graphs.labels;

      cgh.parallel_for<mbsm::device::kernels::RefineDataSignaturesKernel>(global_range, [=](sycl::item<1> item) {
        auto node_id = item.get_id(0);

        uint32_t start_neighbor = row_offsets[node_id];
        uint32_t end_neighbor = row_offsets[node_id + 1];
        mbsm::types::label_t node_label = labels[node_id];

        for (uint32_t i = start_neighbor; i < end_neighbor; ++i) {
          auto neighbor = column_indices[i];
          for (types::label_t l = 0; l < Signature::SignatureDevice::getMaxLabels(); l++) {
            auto count = tmp_buff[neighbor].getLabelCount(l);
            if (l == node_label) { count -= view_size; }
            if (count > 0) signatures[node_id].incrementLabelCount(l, count);
          }
        }
      });
    });
    event.add(refine_event);
    return event;
  }

  template<>
  utils::BatchedEvent refineDataSignatures<Algorithm::PowerGraph>(DeviceBatchedDataGraph& graphs, size_t view_size) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);

    auto refine_event = queue.submit([&](sycl::handler& cgh) {
      auto* row_offsets = graphs.row_offsets;
      auto* column_indices = graphs.column_indices;
      auto* labels = graphs.labels;

      cgh.parallel_for<mbsm::device::kernels::RefineDataSignaturesKernel>(
          global_range, [=, signatures = this->data_signatures, max_labels_count = Signature::SignatureDevice::getMaxLabels()](sycl::item<1> item) {
            auto node_id = item.get_id(0);
            auto graph_id = graphs.getGraphID(node_id);
            auto prev_nodes = graphs.graph_offsets[graph_id];
            utils::detail::Bitset<uint64_t> frontier, reachable;

            frontier.set(node_id - prev_nodes);
            reachable.set(node_id - prev_nodes);
            for (uint curr_iter = 0; curr_iter < view_size && !frontier.empty(); curr_iter++) {
              utils::detail::Bitset<uint64_t> next_frontier;

              for (uint idx = 0; idx < frontier.size(); idx++) {
                auto u = frontier.getSetBit(idx) + prev_nodes;
                auto start_neighbor = row_offsets[u];
                auto end_neighbor = row_offsets[u + 1];
                for (auto i = start_neighbor; i < end_neighbor; ++i) {
                  auto neighbor = column_indices[i] - prev_nodes;
                  if (!reachable.get(neighbor)) {
                    reachable.set(neighbor);
                    next_frontier.set(neighbor);
                  }
                }
                frontier = next_frontier;
              }
            }

            signatures[node_id].clear();
            for (uint idx = 0; idx < reachable.size(); idx++) {
              auto u = reachable.getSetBit(idx) + prev_nodes;
              types::label_t u_label = graphs.labels[u];
              signatures[node_id].incrementLabelCount(u_label);
            }
          });
    });
    event.add(refine_event);
    return event;
  }
};

} // namespace signature
} // namespace mbsm