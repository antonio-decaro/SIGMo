#pragma once
#include "candidates.hpp"
#include "device.hpp"
#include "types.hpp"
#include "utils.hpp"
#include <cstdint>
#include <sycl/sycl.hpp>

namespace mbsm {
namespace signature {

template<size_t Bits = 4>
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
  };


  // TODO consider to use the shared memory to store the graph to avoid uncoallesced memory access
  utils::BatchedEvent generateQuerySignatures(DeviceBatchedQueryGraph& graphs) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);
    auto e = queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<mbsm::device::kernels::GenerateQuerySignaturesKernel<1>>(
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

  utils::BatchedEvent refineQuerySignatures(DeviceBatchedQueryGraph& graphs, size_t iter = 1) {
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

      cgh.parallel_for<mbsm::device::kernels::GenerateQuerySignaturesKernel<2>>(
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
  };

  utils::BatchedEvent generateDataSignatures(DeviceBatchedDataGraph& graphs) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);
    sycl::buffer<mbsm::signature::Signature<>, 1> buffer(sycl::range{global_range});
    auto e = queue.submit([&](sycl::handler& cgh) {
      auto* row_offsets = graphs.row_offsets;
      auto* column_indices = graphs.column_indices;
      auto* labels = graphs.labels;

      cgh.parallel_for<mbsm::device::kernels::GenerateDataSignaturesKernel<1>>(global_range,
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

  utils::BatchedEvent refineDataSignatures(DeviceBatchedDataGraph& graphs, size_t iter = 1) {
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

      cgh.parallel_for<mbsm::device::kernels::GenerateDataSignaturesKernel<2>>(global_range, [=](sycl::item<1> item) {
        auto node_id = item.get_id(0);

        uint32_t start_neighbor = row_offsets[node_id];
        uint32_t end_neighbor = row_offsets[node_id + 1];
        mbsm::types::label_t node_label = labels[node_id];

        for (uint32_t i = start_neighbor; i < end_neighbor; ++i) {
          auto neighbor = column_indices[i];
          for (types::label_t l = 0; l < Signature::SignatureDevice::getMaxLabels(); l++) {
            auto count = tmp_buff[neighbor].getLabelCount(l);
            if (l == node_label) { count -= iter; }
            if (count > 0) signatures[node_id].incrementLabelCount(l, count);
          }
        }
      });
    });
    event.add(refine_event);
    return event;
  }

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
};

} // namespace signature
} // namespace mbsm