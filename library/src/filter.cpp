#include <mbsm.hpp>
#include <sycl/sycl.hpp>

struct Args {
  std::string fname = "/home/adecaro/subgraph-iso-soa/data/MBSM/pool.bin";
  bool print_candidates;

  Args(int& argc, char**& argv) {
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "-p") {
        print_candidates = true;
      } else {
        fname = argv[i];
      }
    }
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

  mbsm::candidates::Signature* data_signatures = sycl::malloc_shared<mbsm::candidates::Signature>(data_nodes, queue);
  std::cout << "Data signatures allocated" << std::endl;
  mbsm::candidates::Signature* query_signatures = sycl::malloc_shared<mbsm::candidates::Signature>(query_nodes, queue);
  std::cout << "Query signatures allocated" << std::endl;

  auto e2 = mbsm::isomorphism::filter::generateDataSignatures(queue, device_data_graph, data_signatures);
  e2.wait();
  auto start = e2.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = e2.get_profiling_info<sycl::info::event_profiling::command_end>();
  std::cout << "Data signatures generated in " << (end - start) * 1e-6 << " ms" << std::endl;
  auto e1 = mbsm::isomorphism::filter::generateQuerySignatures(queue, device_query_graph, query_signatures);
  e1.wait();
  start = e1.get_profiling_info<sycl::info::event_profiling::command_start>();
  end = e1.get_profiling_info<sycl::info::event_profiling::command_end>();
  std::cout << "Query signatures generated in " << (end - start) * 1e-6 << " ms" << std::endl;

  mbsm::candidates::Candidates candidates{query_nodes, data_nodes};
  candidates.candidates = sycl::malloc_shared<mbsm::types::candidates_t>(candidates.getAllocationSize(), queue);
  queue.fill(candidates.candidates, 0, candidates.getAllocationSize()).wait();
  std::cout << "Candidates allocated" << std::endl;

  auto e3 = mbsm::isomorphism::filter::filterCandidates(queue, device_query_graph, device_data_graph, query_signatures, data_signatures, candidates);
  e3.wait();
  start = e3.get_profiling_info<sycl::info::event_profiling::command_start>();
  end = e3.get_profiling_info<sycl::info::event_profiling::command_end>();
  std::cout << "Candidates filtered in " << (end - start) * 1e-6 << " ms" << std::endl;

  if (args.print_candidates) {
    std::cout << "Candidates on " << data_nodes << " data nodes:" << std::endl;

    for (size_t i = 0; i < query_nodes; ++i) {
      auto count = candidates.getCandidatesCount(i, data_nodes);
      std::cout << "Node " << i << ": " << count << std::endl;
    }
  }

  sycl::free(query_signatures, queue);
  sycl::free(data_signatures, queue);
  sycl::free(candidates.candidates, queue);
  mbsm::destroyDeviceDataGraph(device_data_graph, queue);
  mbsm::destroyDeviceQueryGraph(device_query_graph, queue);
}