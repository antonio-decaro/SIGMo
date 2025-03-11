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

  GraphPool(std::vector<mbsm::CSRGraph>& data_graphs, std::vector<mbsm::AMGraph>& query_graphs)
      : _data_graphs(data_graphs), _query_graphs(query_graphs) {}

  std::vector<mbsm::CSRGraph>& getDataGraphs() { return _data_graphs; }

  std::vector<mbsm::AMGraph>& getQueryGraphs() { return _query_graphs; }

  DeviceBatchedCSRGraph transferDataGraphsToDevice(sycl::queue& queue) { return createDeviceCSRGraph(queue, _data_graphs); }

  DeviceBatchedAMGraph transferQueryGraphsToDevice(sycl::queue& queue) { return createDeviceAMGraph(queue, _query_graphs); }

private:
  std::vector<mbsm::CSRGraph> _data_graphs;
  std::vector<mbsm::AMGraph> _query_graphs;
};

} // namespace mbsm
