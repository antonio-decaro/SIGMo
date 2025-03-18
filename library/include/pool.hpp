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

namespace sigmo {

class GraphPool {
public:
  GraphPool() = default;

  GraphPool(std::vector<sigmo::CSRGraph>& data_graphs, std::vector<sigmo::AMGraph>& query_graphs)
      : _data_graphs(data_graphs), _query_graphs(query_graphs) {}

  std::vector<sigmo::CSRGraph>& getDataGraphs() { return _data_graphs; }

  std::vector<sigmo::AMGraph>& getQueryGraphs() { return _query_graphs; }

  DeviceBatchedCSRGraph transferDataGraphsToDevice(sycl::queue& queue) { return createDeviceCSRGraph(queue, _data_graphs); }

  DeviceBatchedAMGraph transferQueryGraphsToDevice(sycl::queue& queue) { return createDeviceAMGraph(queue, _query_graphs); }

private:
  std::vector<sigmo::CSRGraph> _data_graphs;
  std::vector<sigmo::AMGraph> _query_graphs;
};

} // namespace sigmo
