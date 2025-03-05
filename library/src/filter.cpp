#include "./utils.hpp"
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
  size_t num_query_graphs;
  size_t num_data_graphs;

  sycl::queue queue{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
  size_t gpu_mem = queue.get_device().get_info<sycl::info::device::global_mem_size>();
  std::string gpu_name = queue.get_device().get_info<sycl::info::device::name>();

  TimeEvents host_time_events;

  if (args.query_data) {
    auto query_graphs = mbsm::io::loadQueryGraphsFromFile(args.query_file);
    auto data_graphs = mbsm::io::loadDataGraphsFromFile(args.data_file);
    if (args.query_filter.active) {
      for (int i = 0; i < query_graphs.size(); ++i) {
        if (query_graphs[i].getNumNodes() > args.query_filter.max_nodes || query_graphs[i].getNumNodes() < args.query_filter.min_nodes) {
          query_graphs.erase(query_graphs.begin() + i);
          i--;
        }
      }
    }
    num_query_graphs = query_graphs.size();
    for (size_t i = 1; i < args.multiply_factor_query; ++i) {
      query_graphs.insert(query_graphs.end(), query_graphs.begin(), query_graphs.begin() + num_query_graphs);
    }
    num_data_graphs = data_graphs.size();
    for (size_t i = 1; i < args.multiply_factor_data; ++i) {
      data_graphs.insert(data_graphs.end(), data_graphs.begin(), data_graphs.begin() + num_data_graphs);
    }
    device_query_graph = mbsm::createDeviceQueryGraph(queue, query_graphs);
    device_data_graph = mbsm::createDeviceDataGraph(queue, data_graphs);
  } else {
    auto pool = mbsm::io::loadPoolFromBinary(args.fname);
    num_query_graphs = pool.getQueryGraphs().size();
    for (size_t i = 1; i < args.multiply_factor_query; ++i) {
      pool.getQueryGraphs().insert(pool.getQueryGraphs().end(), pool.getQueryGraphs().begin(), pool.getQueryGraphs().begin() + num_query_graphs);
    }
    num_data_graphs = pool.getDataGraphs().size();
    for (size_t i = 1; i < args.multiply_factor_data; ++i) {
      pool.getDataGraphs().insert(pool.getDataGraphs().end(), pool.getDataGraphs().begin(), pool.getDataGraphs().begin() + num_data_graphs);
    }
    device_query_graph = pool.transferQueryGraphsToDevice(queue);
    device_data_graph = pool.transferDataGraphsToDevice(queue);
  }

  size_t data_graph_bytes = mbsm::getDeviceGraphAllocSize(device_data_graph);
  size_t query_graphs_bytes = mbsm::getDeviceGraphAllocSize(device_query_graph);

  std::vector<std::chrono::duration<double>> data_sig_times, query_sig_times, filter_times;

  size_t query_nodes = device_query_graph.total_nodes;
  size_t data_nodes = device_data_graph.total_nodes;

  // get the right filter domain method
  std::function<mbsm::utils::BatchedEvent(
      sycl::queue&, mbsm::DeviceBatchedQueryGraph&, mbsm::DeviceBatchedDataGraph&, mbsm::signature::Signature<>&, mbsm::candidates::Candidates)>
      filter_method, refine_method;
  if (args.isCandidateDomainData()) {
    filter_method = mbsm::isomorphism::filter::filterCandidates<mbsm::candidates::CandidatesDomain::Data>;
    refine_method = mbsm::isomorphism::filter::refineCandidates<mbsm::candidates::CandidatesDomain::Data>;
  } else {
    filter_method = mbsm::isomorphism::filter::filterCandidates<mbsm::candidates::CandidatesDomain::Query>;
    refine_method = mbsm::isomorphism::filter::refineCandidates<mbsm::candidates::CandidatesDomain::Query>;
  }

  std::cout << "------------- Input Data -------------" << std::endl;
  std::cout << "Reed data graph and query graph" << std::endl;
  std::cout << "# Query Nodes " << query_nodes << std::endl;
  std::cout << "# Query Graphs " << num_query_graphs << std::endl;
  std::cout << "# Data Nodes " << data_nodes << std::endl;
  std::cout << "# Data Graphs " << num_data_graphs << std::endl;
  std::cout << "Filter domain: " << args.candidates_domain << std::endl;

  host_time_events.add("setup_data_start");
  std::cout << "------------- Setup Data -------------" << std::endl;
  std::cout << "Allocated " << getBytesSize(data_graph_bytes) << " for graph data" << std::endl;
  std::cout << "Allocated " << getBytesSize(query_graphs_bytes) << " for query data" << std::endl;

  size_t source_nodes = args.isCandidateDomainData() ? data_nodes : query_nodes;
  size_t target_nodes = args.isCandidateDomainData() ? query_nodes : data_nodes;
  mbsm::candidates::Candidates candidates{queue, source_nodes, target_nodes};
  size_t candidates_bytes = candidates.getAllocationSize() * sizeof(mbsm::types::candidates_t);
  std::cout << "Allocated " << getBytesSize(candidates_bytes) << " for candidates" << std::endl;

  mbsm::signature::Signature<> signatures{queue, device_data_graph.total_nodes, device_query_graph.total_nodes};
  size_t data_signatures_bytes = signatures.getDataSignatureAllocationSize();
  std::cout << "Allocated " << getBytesSize(data_signatures_bytes) << " for data signatures" << std::endl;
  size_t query_signatures_bytes = signatures.getQuerySignatureAllocationSize();
  std::cout << "Allocated " << getBytesSize(query_signatures_bytes) << " for query signatures" << std::endl;
  size_t tmp_buff_bytes = std::max(data_signatures_bytes, query_signatures_bytes);
  std::cout << "Allocated " << getBytesSize(tmp_buff_bytes) << " for temporary buffer" << std::endl;
  host_time_events.add("setup_data_end");

  std::cout << "Total allocated memory: "
            << getBytesSize(
                   data_signatures_bytes + query_signatures_bytes + candidates_bytes + tmp_buff_bytes + data_graph_bytes + query_graphs_bytes, false)
            << " out of " << getBytesSize(gpu_mem) << " available on " << gpu_name << std::endl;

  std::cout << "------------- Runtime Filter Phase -------------" << std::endl;
  host_time_events.add("filter_start");
  std::cout << "Initialization Step:" << std::endl;
  auto e1 = signatures.generateDataSignatures(device_data_graph);
  queue.wait_and_throw();
  auto time = e1.getProfilingInfo();
  data_sig_times.push_back(time);
  std::cout << "- Data signatures generated in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

  auto e2 = signatures.generateQuerySignatures(device_query_graph);
  queue.wait_and_throw();
  time = e2.getProfilingInfo();
  query_sig_times.push_back(time);
  std::cout << "- Query signatures generated in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

  auto e3 = filter_method(queue, device_query_graph, device_data_graph, signatures, candidates);
  queue.wait_and_throw();
  time = e3.getProfilingInfo();
  filter_times.push_back(time);
  std::cout << "- Candidates filtered in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

  // start refining candidate set
  for (size_t ref_step = 0; ref_step < args.refinement_steps; ++ref_step) {
    std::cout << "Refinement step " << (ref_step + 1) << ":" << std::endl;

    auto e1 = signatures.refineDataSignatures(device_data_graph, ref_step + 1);
    queue.wait_and_throw();
    time = e1.getProfilingInfo();
    data_sig_times.push_back(time);
    std::cout << "- Data signatures refined in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

    auto e2 = signatures.refineQuerySignatures(device_query_graph, ref_step + 1);
    queue.wait_and_throw();
    time = e2.getProfilingInfo();
    query_sig_times.push_back(time);
    std::cout << "- Query signatures refined in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

    auto e3 = refine_method(queue, device_query_graph, device_data_graph, signatures, candidates);
    queue.wait_and_throw();
    time = e3.getProfilingInfo();
    filter_times.push_back(time);
    std::cout << "- Candidates refined in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;
  }
  host_time_events.add("filter_end");

  if (args.inspect_candidates) {
    CandidatesInspector inspector;
    for (size_t i = 0; i < (args.isCandidateDomainData() ? data_nodes : query_nodes); ++i) {
      auto count = candidates.getCandidatesCount(i);
      inspector.add(count);
      if (args.print_candidates) std::cerr << "Node " << i << ": " << count << std::endl;
    }
    inspector.finalize();
    std::cout << "------------- Results -------------" << std::endl;
    std::cout << "# Total candidates: " << formatNumber(inspector.total) << std::endl;
    std::cout << "# Average candidates: " << formatNumber(inspector.avg) << std::endl;
    std::cout << "# Median candidates: " << formatNumber(inspector.median) << std::endl;
    std::cout << "# Zero candidates: " << formatNumber(inspector.zero_count) << std::endl;
  }

  host_time_events.add("join_start");
  size_t* num_matches = sycl::malloc_shared<size_t>(1, queue);
  *num_matches = 0;
  auto join_e = mbsm::isomorphism::join::joinCandidates(queue, device_query_graph, device_data_graph, candidates, num_matches);
  join_e.wait();
  host_time_events.add("join_end");
  std::cout << "# Matches: " << formatNumber(*num_matches) << std::endl;


  std::cout << "------------- Overall GPU Stats -------------" << std::endl;
  std::chrono::duration<double> total_sig_query_time
      = std::accumulate(query_sig_times.begin(), query_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_sig_data_time = std::accumulate(data_sig_times.begin(), data_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_filter_time = std::accumulate(filter_times.begin(), filter_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_time = total_sig_data_time + total_filter_time + total_sig_query_time + join_e.getProfilingInfo();
  std::cout << "Data signature time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_sig_data_time).count() << " ms" << std::endl;
  std::cout << "Query signature time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_sig_query_time).count() << " ms" << std::endl;
  std::cout << "Filter time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_filter_time).count() << " ms" << std::endl;
  std::cout << "Join time: " << std::chrono::duration_cast<std::chrono::milliseconds>(join_e.getProfilingInfo()).count() << " ms" << std::endl;
  std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() << " ms" << std::endl;

  std::cout << "------------- Overall Host Stats -------------" << std::endl;
  std::cout << "Setup Data time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("setup_data_start", "setup_data_end")).count()
            << " ms" << std::endl;
  std::cout << "Filter time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("filter_start", "filter_end")).count() << " ms"
            << std::endl;
  std::cout << "Join time: " << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("join_start", "join_end")).count()
            << " ms" << std::endl;
  std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getOverallTime()).count() << " ms"
            << std::endl;

  sycl::free(num_matches, queue);
  mbsm::destroyDeviceDataGraph(device_data_graph, queue);
  mbsm::destroyDeviceQueryGraph(device_query_graph, queue);
}