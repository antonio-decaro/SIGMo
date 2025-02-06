#pragma once

#include "host_data.hpp"
#include <sycl/sycl.hpp>

namespace mbsm {
namespace detail {

class SYCL_Graph {
private:
  Graph& graph_data;

  sycl::buffer<size_t, 1> row_offsets;
  sycl::buffer<node_t, 1> col_indices;
  sycl::buffer<label_t, 1> node_labels;

public:
  SYCL_Graph(Graph& graph_data)
      : graph_data(graph_data), row_offsets(graph_data.getRowOffsets().data(), graph_data.getRowOffsets().size()),
        col_indices(graph_data.getColIndices().data(), graph_data.getColIndices().size()),
        node_labels(graph_data.getNodeLabels().data(), graph_data.getNodeLabels().size()) {};

  sycl::accessor<size_t, 1, sycl::access::mode::read> getRowOffsets(sycl::handler& cgh) { return sycl::accessor{row_offsets, cgh, sycl::read_only}; }

  sycl::accessor<node_t, 1, sycl::access::mode::read> getColIndices(sycl::handler& cgh) { return sycl::accessor{col_indices, cgh, sycl::read_only}; }

  sycl::accessor<label_t, 1, sycl::access::mode::read> getNodeLabels(sycl::handler& cgh) { return sycl::accessor{node_labels, cgh, sycl::read_only}; }

  inline const size_t getNumNodes() const { return graph_data.getNumNodes(); }
};

class SYCL_CompressedGraphs {
private:
  CompressedGraphs& graphs_data;
  sycl::buffer<size_t, 1> row_offsets;
  sycl::buffer<node_t, 1> col_indices;
  sycl::buffer<label_t, 1> node_labels;
  sycl::buffer<size_t, 1> offsets;
  sycl::buffer<size_t, 1> sizes;

public:
  SYCL_CompressedGraphs(CompressedGraphs& graphs_data)
      : graphs_data(graphs_data), row_offsets(graphs_data.getRowOffsets().data(), graphs_data.getRowOffsets().size()),
        col_indices(graphs_data.getColIndices().data(), graphs_data.getColIndices().size()),
        node_labels(graphs_data.getNodeLabels().data(), graphs_data.getNodeLabels().size()),
        offsets(graphs_data.getOffsets().data(), graphs_data.getOffsets().size()),
        sizes(graphs_data.getSizes().data(), graphs_data.getSizes().size()) {};

  sycl::accessor<size_t, 1, sycl::access::mode::read> getRowOffsets(sycl::handler& cgh) { return sycl::accessor{row_offsets, cgh, sycl::read_only}; }

  sycl::accessor<node_t, 1, sycl::access::mode::read> getColIndices(sycl::handler& cgh) { return sycl::accessor{col_indices, cgh, sycl::read_only}; }

  sycl::accessor<label_t, 1, sycl::access::mode::read> getNodeLabels(sycl::handler& cgh) { return sycl::accessor{node_labels, cgh, sycl::read_only}; }

  sycl::accessor<size_t, 1, sycl::access::mode::read> getOffsets(sycl::handler& cgh) { return sycl::accessor{offsets, cgh, sycl::read_only}; }

  sycl::accessor<size_t, 1, sycl::access::mode::read> getSizes(sycl::handler& cgh) { return sycl::accessor{sizes, cgh, sycl::read_only}; }

  inline const size_t getNumGraphs() const { return graphs_data.getNumGraphs(); }
};

} // namespace detail

} // namespace mbsm