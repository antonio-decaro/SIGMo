/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "graph.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sycl/sycl.hpp>
#include <vector>

namespace mbsm {

class GraphPool {
public:
  GraphPool() = default;

  GraphPool(std::vector<mbsm::DataGraph>& data_graphs, std::vector<mbsm::QueryGraph>& query_graphs)
      : _data_graphs(data_graphs), _query_graphs(query_graphs) {}

  std::vector<mbsm::DataGraph>& getDataGraphs() { return _data_graphs; }

  std::vector<mbsm::QueryGraph>& getQueryGraphs() { return _query_graphs; }

  DeviceBatchedDataGraph transferDataGraphsToDevice(sycl::queue& queue) { return createDeviceDataGraph(queue, _data_graphs); }

  DeviceBatchedQueryGraph transferQueryGraphsToDevice(sycl::queue& queue) { return createDeviceQueryGraph(queue, _query_graphs); }

private:
  std::vector<mbsm::DataGraph> _data_graphs;
  std::vector<mbsm::QueryGraph> _query_graphs;
};

} // namespace mbsm
