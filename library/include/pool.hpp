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

  void transferDataGraphsToDevice(sycl::queue& queue) {
    for (auto& graph : _data_graphs) {
      // TODO Implement transfer logic for DataGraph
    }
  }

  void transferQueryGraphsToDevice(sycl::queue& queue) {
    for (auto& graph : _query_graphs) {
      // TODO Implement transfer logic for QueryGraph
    }
  }

  void transferDataGraphsToDeviceAsync(sycl::queue& queue, std::vector<sycl::event>& events) {
    for (auto& graph : _data_graphs) {
      // TODO Implement asynchronous transfer logic for DataGraph
      sycl::event event; // Placeholder for actual event
      events.push_back(event);
    }
  }

  void transferQueryGraphsToDeviceAsync(sycl::queue& queue, std::vector<sycl::event>& events) {
    for (auto& graph : _query_graphs) {
      // TODO Implement asynchronous transfer logic for QueryGraph
      sycl::event event; // Placeholder for actual event
      events.push_back(event);
    }
  }

private:
  std::vector<mbsm::DataGraph> _data_graphs;
  std::vector<mbsm::QueryGraph> _query_graphs;
};

} // namespace mbsm
