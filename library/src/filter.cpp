#include "../../library/test/include/data.hpp"
#include "./arg_parse.hpp"
#include <mbsm.hpp>
#include <sycl/sycl.hpp>

struct CandidatesInspector {
  std::vector<size_t> candidates_sizes;
  size_t total = 0;
  size_t avg = 0;
  size_t median = 0;
  size_t zero_count = 0;

  void add(size_t size) { candidates_sizes.push_back(size); }

  void finalize() {
    for (auto& size : candidates_sizes) {
      total += size;
      if (size == 0) zero_count++;
    }

    avg = total / candidates_sizes.size();
    std::sort(candidates_sizes.begin(), candidates_sizes.end());
    median = candidates_sizes[candidates_sizes.size() / 2];
  }
};

int main(int argc, char** argv) {
  Args args{argc, argv};

  mbsm::DeviceBatchedDataGraph device_data_graph;
  mbsm::DeviceBatchedQueryGraph device_query_graph;
  sycl::queue queue{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};

  if (args.query_data) {
    auto query_graphs = mbsm::io::loadQueryGraphsFromFile(args.query_file);
    auto data_graphs = mbsm::io::loadDataGraphsFromFile(args.data_file);
    device_query_graph = mbsm::createDeviceQueryGraph(queue, query_graphs);
    device_data_graph = mbsm::createDeviceDataGraph(queue, data_graphs);
  } else {
    auto pool = mbsm::io::loadPoolFromBinary(args.fname);
    device_query_graph = pool.transferQueryGraphsToDevice(queue);
    device_data_graph = pool.transferDataGraphsToDevice(queue);
  }

  size_t query_nodes = device_query_graph.total_nodes;
  size_t data_nodes = device_data_graph.total_nodes;

  std::cout << "Reed data graph and query graph" << std::endl;

  mbsm::candidates::Candidates candidates{query_nodes, data_nodes};
  candidates.candidates = sycl::malloc_shared<mbsm::types::candidates_t>(candidates.getAllocationSize(), queue);
  queue.fill(candidates.candidates, 0, candidates.getAllocationSize()).wait();
  std::cout << "Candidates allocated and initialized" << std::endl;

  mbsm::signature::Signature<>* data_signatures = sycl::malloc_shared<mbsm::signature::Signature<>>(data_nodes, queue);
  queue.fill(data_signatures, 0, data_nodes).wait();
  std::cout << "Data signatures allocated" << std::endl;
  mbsm::signature::Signature<>* query_signatures = sycl::malloc_shared<mbsm::signature::Signature<>>(query_nodes, queue);
  queue.fill(query_signatures, 0, query_nodes).wait();
  std::cout << "Query signatures allocated" << std::endl;
  mbsm::signature::Signature<>* tmp_buff = sycl::malloc_shared<mbsm::signature::Signature<>>(std::max(query_nodes, data_nodes), queue);
  std::cout << "Temporary buffer allocated" << std::endl;

  auto e1 = mbsm::signature::generateDataSignatures(queue, device_data_graph, data_signatures);
  queue.wait_and_throw();
  auto time = e1.getProfilingInfo();
  std::cout << "Data signatures generated in " << std::chrono::duration_cast<std::chrono::microseconds>(time).count() << " us" << std::endl;

  auto e2 = mbsm::signature::generateQuerySignatures(queue, device_query_graph, query_signatures);
  queue.wait_and_throw();
  time = e2.getProfilingInfo();
  std::cout << "Query signatures generated in " << std::chrono::duration_cast<std::chrono::microseconds>(time).count() << " us" << std::endl;

  auto e3 = mbsm::isomorphism::filter::filterCandidates(queue, device_query_graph, device_data_graph, query_signatures, data_signatures, candidates);
  queue.wait_and_throw();
  time = e3.getProfilingInfo();
  std::cout << "Candidates filtered in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

  // start refining candidate set
  for (size_t ref_step = 0; ref_step < args.refinement_steps; ++ref_step) {
    std::cout << "-----------------------" << std::endl;
    std::cout << "Refinement step: " << (ref_step + 1) << std::endl;
    std::cout << "-----------------------" << std::endl;

    auto e1 = mbsm::signature::refineDataSignatures(queue, device_data_graph, data_signatures, tmp_buff);
    queue.wait_and_throw();
    time = e1.getProfilingInfo();
    std::cout << "Data signatures refined in " << std::chrono::duration_cast<std::chrono::microseconds>(time).count() << " us" << std::endl;

    auto e2 = mbsm::signature::refineQuerySignatures(queue, device_query_graph, query_signatures, tmp_buff);
    queue.wait_and_throw();
    time = e2.getProfilingInfo();
    std::cout << "Query signatures refined in " << std::chrono::duration_cast<std::chrono::microseconds>(time).count() << " us" << std::endl;

    auto e3
        = mbsm::isomorphism::filter::refineCandidates(queue, device_query_graph, device_data_graph, query_signatures, data_signatures, candidates);
    queue.wait_and_throw();
    time = e3.getProfilingInfo();
    std::cout << "Candidates refined in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;
  }
  std::cout << "Candidates of " << query_nodes << " on " << data_nodes << " data nodes" << std::endl;

  CandidatesInspector inspector;
  for (size_t i = 0; i < query_nodes; ++i) {
    auto count = candidates.getCandidatesCount(i);
    inspector.add(count);
    if (args.print_candidates) std::cerr << "Node " << i << ": " << count << std::endl;
  }
  inspector.finalize();
  std::cout << "Info:" << std::endl;
  std::cout << "- Total candidates: " << formatNumber(inspector.total) << std::endl;
  std::cout << "- Average candidates: " << formatNumber(inspector.avg) << std::endl;
  std::cout << "- Median candidates: " << formatNumber(inspector.median) << std::endl;
  std::cout << "- Zero candidates: " << formatNumber(inspector.zero_count) << std::endl;

  sycl::free(tmp_buff, queue);
  sycl::free(query_signatures, queue);
  sycl::free(data_signatures, queue);
  sycl::free(candidates.candidates, queue);
  mbsm::destroyDeviceDataGraph(device_data_graph, queue);
  mbsm::destroyDeviceQueryGraph(device_query_graph, queue);
}