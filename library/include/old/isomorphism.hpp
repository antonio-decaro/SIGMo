#pragma once

#include "host_data.hpp"
#include "sycl_data.hpp"
#include "sycl_utils.hpp"
#include "types.hpp"
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>

namespace mbsm {
using namespace detail;

class SubgraphIsomorphism {
private:
  sycl::queue& queue;
  Graph& data_graph;
  CompressedGraphs compressed_query_graphs;
  SYCL_Graph sycl_data_graph;
  SYCL_CompressedGraphs sycl_query_graphs;

  std::vector<size_t> query_offsets;
  std::shared_ptr<BitmaskMap> candidates;

  std::future<bool> final_event;
  std::chrono::duration<double> time;
  std::vector<std::vector<std::pair<node_t, node_t>>> answare;

protected:
  template<size_t sg_size = 16>
  sycl::event pruneVertices(size_t max_local_size) {
    size_t num_query_graphs = compressed_query_graphs.getNumGraphs();
    auto local_size = std::min(max_local_size, sg_size * num_query_graphs);

    auto e = queue.submit([&](sycl::handler& cgh) {
      auto query_row_offsets = sycl_query_graphs.getRowOffsets(cgh);
      auto query_col_indices = sycl_query_graphs.getColIndices(cgh);
      auto query_labels = sycl_query_graphs.getNodeLabels(cgh);
      auto query_offsets = sycl_query_graphs.getOffsets(cgh);
      auto query_graph_sizes = sycl_query_graphs.getSizes(cgh);

      auto data_row_offsets = sycl_data_graph.getRowOffsets(cgh);
      auto data_col_indices = sycl_data_graph.getColIndices(cgh);
      auto data_labels = sycl_data_graph.getNodeLabels(cgh);

      auto candidates = this->candidates->getDeviceAccessor(cgh);

      sycl::range local{local_size};
      sycl::range global{local_size};

      cgh.parallel_for<class PruneVertices>(
          sycl::nd_range<1>{global, local},
          [=,
           n_query = this->sycl_query_graphs.getNumGraphs(),
           mask_size = MASK_SIZE,
           arrmask_lenght = this->candidates->getSingleMaskLenght(),
           data_size = sycl_data_graph.getNumNodes()](sycl::nd_item<1> id) [[intel::reqd_sub_group_size(sg_size)]]
          {
            auto local_size = id.get_local_range()[0];
            auto sg = id.get_sub_group();
            auto sg_id = sg.get_group_id();
            auto lid = sg.get_local_id();
            auto n_sg = local_size / sg_size;

            // iterate over the query_graphs to increase the utilization
            for (int query_graph = sg_id; query_graph < n_query; query_graph += n_sg) {
              auto query_graph_size = query_graph_sizes[query_graph];
              auto offset = query_offsets[query_graph];

              for (int data_node = lid; data_node < data_size; data_node += sg_size) {
                auto data_node_label = data_labels[data_node];

                auto data_node_begin = data_row_offsets[data_node];
                auto data_node_end = data_row_offsets[data_node + 1];

                // prune vertices based on the label
                for (auto query_node = 0; query_node < query_graph_size; query_node++) {
                  auto mask = mbsm::device::bitmask::getAtomicRef(candidates, offset + query_node, data_node, arrmask_lenght, mask_size);
                  auto query_node_label = query_labels[offset + query_node];
                  if (mbsm::device::graph::labelMatch(query_node_label, data_node_label)) {
                    auto query_node_begin = query_row_offsets[offset + query_node];
                    auto query_node_end = query_row_offsets[offset + query_node + 1];

                    // If the query node has more neighbours than the data node, then it cannot be a candidate
                    if (query_node_end - query_node_begin > data_node_end - data_node_begin) {
                      mbsm::device::bitmask::setOff(mask, data_node, mask_size);
                    }
                  } else {
                    mbsm::device::bitmask::setOff(mask, data_node, mask_size);
                  }
                }

                sycl::group_barrier(sg, sycl::memory_scope::sub_group);

                // prune vertices based on the neighbourhood
                for (auto query_node = 0; query_node < query_graph_size; query_node++) {
                  auto mask = mbsm::device::bitmask::getAtomicRef(candidates, offset + query_node, data_node, arrmask_lenght, mask_size);
                  if (mbsm::device::bitmask::get(candidates, offset + query_node, data_node, arrmask_lenght, mask_size)) {
                    if (!mbsm::device::graph::isIsomorphic(query_node,
                                                           data_node,
                                                           offset,
                                                           data_row_offsets,
                                                           query_row_offsets,
                                                           data_col_indices,
                                                           query_col_indices,
                                                           data_labels,
                                                           query_labels,
                                                           candidates,
                                                           arrmask_lenght,
                                                           mask_size)) {
                      mbsm::device::bitmask::setOff(mask, data_node, mask_size);
                    }
                  }
                }
              }
            }
          });
    });

    return e;
  }

  std::vector<std::pair<node_t, node_t>> findSolutions(size_t graph_id) {
    const Graph& query_graph = compressed_query_graphs.getGraphs()[graph_id];
    const size_t offset = query_offsets[graph_id];

    auto backtrack = [=](auto&& self, node_t query_node, node_t data_node, std::set<node_t> query_visited, std::set<node_t>& data_visited)
        -> std::vector<std::pair<node_t, node_t>> {
      std::vector<std::pair<node_t, node_t>> res;

      query_visited.insert(query_node);
      data_visited.insert(data_node);

      std::vector<node_t> unvisited_neighbours;

      for (auto& query_neighbour : query_graph.getNeighbours(query_node)) {
        if (query_visited.find(query_neighbour) == query_visited.end()) { unvisited_neighbours.push_back(query_neighbour); }
      }
      if (unvisited_neighbours.empty()) { return {{query_node, data_node}}; }

      for (auto& query_neighbour : unvisited_neighbours) {
        auto potential_mapping = candidates->getOnNodes(offset + query_neighbour);
        if (potential_mapping.empty()) { return {}; }
        std::vector<std::pair<node_t, node_t>> neighbour_res;

        for (auto& data_neighbour : potential_mapping) {
          if (data_visited.find(data_neighbour) != data_visited.end()) { continue; }
          if (data_graph.isNeighbour(data_node, data_neighbour)) {
            auto tmp = self(self, query_neighbour, data_neighbour, query_visited, data_visited);
            if (!tmp.empty()) {
              neighbour_res.insert(neighbour_res.end(), tmp.begin(), tmp.end());
              break;
            }
          }
        }
        if (neighbour_res.empty()) { return {}; }
        res.insert(res.end(), neighbour_res.begin(), neighbour_res.end());
        for (auto& [query_node, data_node] : neighbour_res) { data_visited.insert(data_node); }
      }
      if (res.empty()) {
        data_visited.erase(data_node);
        return {};
      }
      res.push_back({query_node, data_node});
      return res;
    };

    size_t min_candidates;
    node_t start_node = 0;
    for (int i = 0; i < query_graph.getNumNodes(); i++) {
      auto tmp = candidates->getOnNodes(offset + i);
      if (tmp.size() < min_candidates) {
        min_candidates = tmp.size();
        start_node = i;
      }
    }
    std::vector<std::pair<node_t, node_t>> res;
    auto start_node_candidates = candidates->getOnNodes(offset + start_node);
    for (auto& start_node_candidate : start_node_candidates) {
      std::set<node_t> data_visited;
      auto tmp = backtrack(backtrack, start_node, start_node_candidate, {}, data_visited);
      if (!tmp.empty()) {
        res.insert(res.end(), tmp.begin(), tmp.end());
        break;
      }
    }
    return res;
  }

  // TODO: Transoform it to a kernel
  void combinations(node_t node, std::vector<node_t> path = {}) {
    //   if (node == query_graph.getNumNodes()) {
    //     res.insert(res.end(), path.begin(), path.end());
    //     return;
    //   }
    //   for (auto candidate : candidates->getOnNodes(node)) {
    //     if (std::find(path.begin(), path.end(), candidate) == path.end()) {
    //       path.push_back(candidate);
    //       combinations(node + 1, path);
    //       path.pop_back();
    //     }
    //   }
  }

  std::vector<size_t> getDFSEdges(const Graph& g, const node_t start_node) {
    std::vector<size_t> edges;
    std::vector<bool> visited(g.getNumNodes(), false);
    std::function<void(node_t)> dfs = [&](node_t node) {
      visited[node] = true;
      for (size_t i = g.getRowOffsets()[node]; i < g.getRowOffsets()[node + 1]; i++) {
        auto neighbour = g.getColIndices()[i];
        if (!visited[neighbour]) {
          edges.push_back(i);
          dfs(neighbour);
        }
      }
    };
    dfs(start_node);
    return edges;
  }

  std::vector<std::tuple<node_t, node_t>> getDFSTupleEdges(const Graph& g, const node_t start_node) {
    std::vector<std::tuple<node_t, node_t>> edges;
    std::vector<bool> visited(g.getNumNodes(), false);
    std::function<void(node_t)> dfs = [&](node_t node) {
      visited[node] = true;
      for (auto neighbour : g.getNeighbours(node)) {
        if (!visited[neighbour]) {
          edges.push_back({node, neighbour});
          dfs(neighbour);
        }
      }
    };
    dfs(start_node);
    return edges;
  }

  void printCandidates() {
    for (int i = 0; i < compressed_query_graphs.getNumGraphs(); i++) {
      auto offset = compressed_query_graphs.getOffsets()[i];
      auto graph_size = compressed_query_graphs.getSizes()[i];

      std::cout << "Query graph " << i << ": " << std::endl;
      for (auto j = 0; j < graph_size; j++) {
        auto onNodes = candidates->getOnNodes(offset + j);
        std::vector<node_t> on;
        std::cout << "\t- " << j << ": [";
        for (auto n : onNodes) { std::cout << n << ", "; }
        std::cout << "]" << std::endl;
        std::cout << std::endl;
      }
    }
  }

  template<size_t sg_size = 16>
  sycl::event pruneEdges(size_t max_local_size) {
    size_t num_query_graphs = compressed_query_graphs.getNumGraphs();
    auto local_size = std::min(max_local_size, sg_size * num_query_graphs);
    sycl::range local{local_size};
    sycl::range global{local_size};

    std::vector<size_t> candidates_offset;
    std::vector<node_t> candidates_indices;

    size_t offset = 0;
    for (auto curr_node = 0; curr_node < compressed_query_graphs.getRowOffsets().size() - 1; curr_node++) {
      auto curr_candidates = candidates->getOnNodes(curr_node);
      candidates_offset.push_back(offset);
      candidates_indices.insert(candidates_indices.end(), curr_candidates.begin(), curr_candidates.end());
      offset += curr_candidates.size();
    }
    candidates_offset.push_back(offset);

    sycl::buffer<size_t, 1> candidates_offset_buffer{candidates_offset.data(), candidates_offset.size()};
    sycl::buffer<node_t, 1> candidates_indices_buffer{candidates_indices.data(), candidates_indices.size()};

    return queue.submit([&](sycl::handler& cgh) {
      sycl::accessor candidates_offset{candidates_offset_buffer, cgh, sycl::read_only};
      sycl::accessor candidates_indices{candidates_indices_buffer, cgh, sycl::read_only};

      auto query_row_offsets = sycl_query_graphs.getRowOffsets(cgh);
      auto query_col_indices = sycl_query_graphs.getColIndices(cgh);
      auto query_labels = sycl_query_graphs.getNodeLabels(cgh);
      auto query_offsets = sycl_query_graphs.getOffsets(cgh);
      auto query_graph_sizes = sycl_query_graphs.getSizes(cgh);

      auto data_row_offsets = sycl_data_graph.getRowOffsets(cgh);
      auto data_col_indices = sycl_data_graph.getColIndices(cgh);
      auto data_labels = sycl_data_graph.getNodeLabels(cgh);

      auto candidates = this->candidates->getDeviceAccessor(cgh);

      using frontier_t = uint64_t;
      const size_t frontier_size = sizeof(frontier_t) * 8;
      sycl::local_accessor<frontier_t, 1> frontier_maps{this->sycl_query_graphs.getNumGraphs(), cgh};
      sycl::local_accessor<frontier_t, 1> next_maps{this->sycl_query_graphs.getNumGraphs(), cgh};
      sycl::local_accessor<frontier_t, 1> visited_maps{this->sycl_query_graphs.getNumGraphs(), cgh};

      sycl::stream out(1024, 256, cgh);

      cgh.parallel_for<class PruneEdges>(
          sycl::nd_range<1>{global, local},
          [=,
           n_query = this->sycl_query_graphs.getNumGraphs(),
           mask_size = MASK_SIZE,
           arrmask_lenght = this->candidates->getSingleMaskLenght(),
           data_size = sycl_data_graph.getNumNodes()](sycl::nd_item<1> item) {
            auto local_size = item.get_local_range()[0];
            auto sg = item.get_sub_group();
            auto sg_id = sg.get_group_id();
            auto lid = sg.get_local_id();
            auto n_sg = local_size / sg_size;

            for (int query_graph = sg_id; query_graph < n_query; query_graph += n_sg) {
              auto query_graph_size = query_graph_sizes[query_graph];
              auto offset = query_offsets[query_graph];
              sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> frontier(frontier_maps[query_graph]);
              sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> next(next_maps[query_graph]);
              sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> visited{visited_maps[query_graph]};

              if (lid == 0) { visited = frontier = next = 1; }

              sycl::group_barrier(sg, sycl::memory_scope::sub_group);
              while (next) {
                sycl::group_barrier(sg, sycl::memory_scope::sub_group);
                if (lid == 0) {
                  frontier.store(next.load());
                  next = 0;
                }
                sycl::group_barrier(sg, sycl::memory_scope::sub_group);

                for (node_t query_node = lid; query_node < query_graph_size; query_node += sg_size) {
                  frontier_t node_bit = 1 << query_node;
                  // compute next frontier
                  if (!(visited & node_bit)) {
                    for (auto i = query_row_offsets[offset + query_node]; i < query_row_offsets[offset + query_node + 1]; i++) {
                      node_t query_neighbour = query_col_indices[i];
                      frontier_t neighbour_bit = 1 << query_neighbour;
                      if (frontier & neighbour_bit) {
                        visited |= node_bit;
                        next |= node_bit;

                        for (int c2 = candidates_offset[offset + query_neighbour]; c2 < candidates_offset[offset + query_neighbour + 1]; c2++) {
                          node_t data2 = candidates_indices[c2];
                          bool found = false;
                          for (int c1 = candidates_offset[offset + query_node]; c1 < candidates_offset[offset + query_node + 1]; c1++) {
                            node_t data1 = candidates_indices[c1];
                            if (mbsm::device::graph::isNeighbour(data1, data2, data_row_offsets, data_col_indices)) {
                              found = true;
                              break;
                            }
                          }
                          if (!found) { mbsm::device::bitmask::setOff(candidates, offset + query_neighbour, data2, arrmask_lenght, mask_size); }
                        }

                        break;
                      }
                    }
                  }
                }
                sycl::group_barrier(sg, sycl::memory_scope::sub_group);
              }
            }
          });
    });
  }

public:
  SubgraphIsomorphism(Graph& data_graph, std::vector<Graph>& query_graphs, sycl::queue& queue)
      : data_graph(data_graph), compressed_query_graphs(query_graphs), queue(queue), sycl_query_graphs(compressed_query_graphs),
        sycl_data_graph(data_graph) {
    size_t offset = 0;

    for (auto& query : query_graphs) {
      query_offsets.push_back(offset);
      offset += query.getNumNodes();
    }
    answare.resize(compressed_query_graphs.getNumGraphs());
    candidates = std::make_shared<BitmaskMap>(BitmaskMap(offset, data_graph.getNumNodes(), true));
  };

  void run() {
    final_event = std::move(std::async(std::launch::async, [&]() {
      try {
        auto max_local_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
        auto start = std::chrono::high_resolution_clock::now();
        auto prune_vertices = pruneVertices(max_local_size);
        prune_vertices.wait_and_throw();
        auto prune_edges = pruneEdges(max_local_size);
        prune_edges.wait_and_throw();
        std::vector<std::future<void>> futures;
        for (int i = 0; i < compressed_query_graphs.getNumGraphs(); i++) {
          futures.push_back(std::async(std::launch::async, [&, i]() {
            auto res = findSolutions(i);
            answare[i] = res;
          }));
        }
        for (auto& f : futures) { f.wait(); }
        auto end = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return true;
      } catch (sycl::exception& e) { std::cerr << e.what() << std::endl; } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
      } catch (...) { std::cerr << "Unknown error" << std::endl; }
      return false;
    }));
  }

  void wait() {
    if (final_event.valid()) {
      if (!final_event.get()) { throw std::runtime_error("Event failed"); }
    } else {
      throw std::runtime_error("Event not initialized");
    }
  }

  void wait_for(std::chrono::seconds time) {
    if (final_event.valid()) {
      if (final_event.wait_for(time) == std::future_status::timeout) {
        throw std::runtime_error("Timeout");
      } else {
        if (!final_event.get()) { throw std::runtime_error("Event failed"); }
      }
    } else {
      throw std::runtime_error("Event not initialized");
    }
  }

  [[deprecated("Use getAnswers() instead")]]
  const BitmaskMap& getCandidates() {
    return *candidates;
  }

  // TODO
  const std::vector<std::vector<std::pair<node_t, node_t>>> getAnswers() const { return answare; }


  /**
   * @brief Returns the execution time (us) of the function.
   *
   * @return The execution time in microseconds.
   */
  inline const double getExceutionTime() const { return time.count(); }
};

} // namespace mbsm