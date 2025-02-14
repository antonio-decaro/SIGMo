#include "./arg_parse.hpp"
#include <mbsm.hpp>
#include <sycl/sycl.hpp>

struct CandidatesInspector {
  std::vector<size_t> candidates_sizes;
  size_t total;
  size_t avg;
  size_t median;
  size_t zero_count;

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

  auto pool = mbsm::io::loadPoolFromBinary(args.fname);

  sycl::queue queue{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};

  auto device_query_graph = mbsm::createDeviceQueryGraph(queue, pool.getQueryGraphs());
  auto device_data_graph = mbsm::createDeviceDataGraph(queue, pool.getDataGraphs());
  size_t query_nodes = device_query_graph.total_nodes;
  size_t data_nodes = device_data_graph.total_nodes;

  std::cout << "Reed data graph and query graph" << std::endl;

  mbsm::candidates::Candidates curr_candidates{query_nodes, data_nodes};
  mbsm::candidates::Candidates prev_candidates{query_nodes, data_nodes};
  curr_candidates.candidates = sycl::malloc_shared<mbsm::types::candidates_t>(curr_candidates.getAllocationSize(), queue);
  prev_candidates.candidates = sycl::malloc_shared<mbsm::types::candidates_t>(prev_candidates.getAllocationSize(), queue);
  queue.fill(curr_candidates.candidates, 0, curr_candidates.getAllocationSize());
  queue.fill(prev_candidates.candidates, 0, prev_candidates.getAllocationSize());
  queue.wait_and_throw();
  std::cout << "Candidates allocated" << std::endl;

  for (size_t ref_step = 0; ref_step < args.refinement_steps; ++ref_step) {
    std::cout << "-----------------------" << std::endl;
    std::cout << "Filtering step: " << ref_step << std::endl;
    std::cout << "-----------------------" << std::endl;

    mbsm::candidates::Signature<>* data_signatures = sycl::malloc_shared<mbsm::candidates::Signature<>>(data_nodes, queue);
    std::cout << "Data signatures allocated" << std::endl;
    mbsm::candidates::Signature<>* query_signatures = sycl::malloc_shared<mbsm::candidates::Signature<>>(query_nodes, queue);
    std::cout << "Query signatures allocated" << std::endl;
    auto gds_e = mbsm::isomorphism::filter::generateDataSignatures(queue, device_data_graph, data_signatures, ref_step);
    gds_e.wait();
    auto time = gds_e.getProfilingInfo();
    std::cout << "Data signatures generated in " << std::chrono::duration_cast<std::chrono::microseconds>(time).count() << " us" << std::endl;
    auto gqs_e = mbsm::isomorphism::filter::generateQuerySignatures(queue, device_query_graph, query_signatures, ref_step);
    gqs_e.wait();
    time = gqs_e.getProfilingInfo();
    std::cout << "Query signatures generated in " << std::chrono::duration_cast<std::chrono::microseconds>(time).count() << " us" << std::endl;

    mbsm::utils::BatchedEvent filter_e;
    if (ref_step == 0) {
      filter_e = mbsm::isomorphism::filter::filterCandidates(
          queue, device_query_graph, device_data_graph, query_signatures, data_signatures, curr_candidates);
    } else {
      filter_e = mbsm::isomorphism::filter::filterCandidates(
          queue, device_query_graph, device_data_graph, query_signatures, data_signatures, curr_candidates, prev_candidates);
    }
    filter_e.wait();
    time = filter_e.getProfilingInfo();
    std::cout << "Candidates filtered in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

    queue.copy(curr_candidates.candidates, prev_candidates.candidates, curr_candidates.getAllocationSize()).wait();
    if (ref_step + 1 < args.refinement_steps) { queue.fill(curr_candidates.candidates, 0, curr_candidates.getAllocationSize()).wait(); }

    sycl::free(query_signatures, queue);
    sycl::free(data_signatures, queue);
  }
  std::cout << "-----------------------------" << std::endl;
  std::cout << "Candidates of " << query_nodes << " on " << data_nodes << " data nodes" << std::endl;

  CandidatesInspector inspector;
  for (size_t i = 0; i < query_nodes; ++i) {
    auto count = curr_candidates.getCandidatesCount(i, data_nodes);
    inspector.add(count);
    if (args.print_candidates) std::cerr << "Node " << i << ": " << count << std::endl;
  }
  inspector.finalize();
  std::cout << "Info:" << std::endl;
  std::cout << "- Total candidates: " << formatNumber(inspector.total) << std::endl;
  std::cout << "- Average candidates: " << formatNumber(inspector.avg) << std::endl;
  std::cout << "- Median candidates: " << formatNumber(inspector.median) << std::endl;
  std::cout << "- Zero candidates: " << formatNumber(inspector.zero_count) << std::endl;

  sycl::free(prev_candidates.candidates, queue);
  sycl::free(curr_candidates.candidates, queue);
  mbsm::destroyDeviceDataGraph(device_data_graph, queue);
  mbsm::destroyDeviceQueryGraph(device_query_graph, queue);
}