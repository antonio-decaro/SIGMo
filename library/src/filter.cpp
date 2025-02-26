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

  std::cout << "------------- Input Data -------------" << std::endl;
  std::cout << "Reed data graph and query graph" << std::endl;
  std::cout << "# Query Nodes " << query_nodes << std::endl;
  std::cout << "# Query Graphs " << num_query_graphs << std::endl;
  std::cout << "# Data Nodes " << data_nodes << std::endl;
  std::cout << "# Data Graphs " << num_data_graphs << std::endl;

  host_time_events.add("setup_data_start");
  std::cout << "------------- Setup Data -------------" << std::endl;
  std::cout << "Allocated " << getBytesSize(data_graph_bytes) << " for graph data" << std::endl;
  std::cout << "Allocated " << getBytesSize(query_graphs_bytes) << " for query data" << std::endl;

  mbsm::candidates::Candidates candidates{data_nodes, query_nodes};
  size_t alloc_size = candidates.getAllocationSize();
  candidates.candidates = sycl::malloc_shared<mbsm::types::candidates_t>(alloc_size, queue);
  queue.fill(candidates.candidates, 0, alloc_size).wait();
  size_t candidates_bytes = alloc_size * sizeof(mbsm::types::candidates_t);
  std::cout << "Allocated " << getBytesSize(candidates_bytes) << " for candidates" << std::endl;

  mbsm::signature::Signature<>* data_signatures = sycl::malloc_shared<mbsm::signature::Signature<>>(data_nodes, queue);
  queue.fill(data_signatures, 0, data_nodes).wait();
  size_t data_signatures_bytes = data_nodes * sizeof(mbsm::signature::Signature<>);
  std::cout << "Allocated " << getBytesSize(data_signatures_bytes) << " for data signatures" << std::endl;

  mbsm::signature::Signature<>* query_signatures = sycl::malloc_shared<mbsm::signature::Signature<>>(query_nodes, queue);
  queue.fill(query_signatures, 0, query_nodes).wait();
  size_t query_signatures_bytes = query_nodes * sizeof(mbsm::signature::Signature<>);
  std::cout << "Allocated " << getBytesSize(query_signatures_bytes) << " for query signatures" << std::endl;

  mbsm::signature::Signature<>* tmp_buff = sycl::malloc_shared<mbsm::signature::Signature<>>(std::max(query_nodes, data_nodes), queue);
  size_t tmp_buff_bytes = std::max(query_nodes, data_nodes) * sizeof(mbsm::signature::Signature<>);
  std::cout << "Allocated " << getBytesSize(tmp_buff_bytes) << " for temporary buffer" << std::endl;
  host_time_events.add("setup_data_end");

  std::cout << "Total allocated memory: "
            << getBytesSize(
                   data_signatures_bytes + query_signatures_bytes + candidates_bytes + tmp_buff_bytes + data_graph_bytes + query_graphs_bytes, false)
            << " out of " << getBytesSize(gpu_mem) << " available on " << gpu_name << std::endl;

  std::cout << "------------- Runtime Filter Phase -------------" << std::endl;
  host_time_events.add("filter_start");
  std::cout << "Initialization Step:" << std::endl;
  auto e1 = mbsm::signature::generateDataSignatures(queue, device_data_graph, data_signatures);
  queue.wait_and_throw();
  auto time = e1.getProfilingInfo();
  data_sig_times.push_back(time);
  std::cout << "- Data signatures generated in " << std::chrono::duration_cast<std::chrono::microseconds>(time).count() << " us" << std::endl;

  auto e2 = mbsm::signature::generateQuerySignatures(queue, device_query_graph, query_signatures);
  queue.wait_and_throw();
  time = e2.getProfilingInfo();
  query_sig_times.push_back(time);
  std::cout << "- Query signatures generated in " << std::chrono::duration_cast<std::chrono::microseconds>(time).count() << " us" << std::endl;

  auto e3 = mbsm::isomorphism::filter::filterCandidates(queue, device_query_graph, device_data_graph, query_signatures, data_signatures, candidates);
  queue.wait_and_throw();
  time = e3.getProfilingInfo();
  filter_times.push_back(time);
  std::cout << "- Candidates filtered in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

  // start refining candidate set
  for (size_t ref_step = 0; ref_step < args.refinement_steps; ++ref_step) {
    std::cout << "Refinement step " << (ref_step + 1) << ":" << std::endl;

    auto e1 = mbsm::signature::refineDataSignatures(queue, device_data_graph, data_signatures, tmp_buff, ref_step + 1);
    queue.wait_and_throw();
    time = e1.getProfilingInfo();
    data_sig_times.push_back(time);
    std::cout << "- Data signatures refined in " << std::chrono::duration_cast<std::chrono::microseconds>(time).count() << " us" << std::endl;

    auto e2 = mbsm::signature::refineQuerySignatures(queue, device_query_graph, query_signatures, tmp_buff, ref_step + 1);
    queue.wait_and_throw();
    time = e2.getProfilingInfo();
    query_sig_times.push_back(time);
    std::cout << "- Query signatures refined in " << std::chrono::duration_cast<std::chrono::microseconds>(time).count() << " us" << std::endl;

    auto e3
        = mbsm::isomorphism::filter::refineCandidates(queue, device_query_graph, device_data_graph, query_signatures, data_signatures, candidates);
    queue.wait_and_throw();
    time = e3.getProfilingInfo();
    filter_times.push_back(time);
    std::cout << "- Candidates refined in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;
  }
  host_time_events.add("filter_end");

  std::cout << "------------- Overall GPU Stats -------------" << std::endl;
  std::chrono::duration<double> total_sig_query_time
      = std::accumulate(query_sig_times.begin(), query_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_sig_data_time = std::accumulate(data_sig_times.begin(), data_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_filter_time = std::accumulate(filter_times.begin(), filter_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_time = total_sig_data_time + total_filter_time + total_sig_query_time;
  std::cout << "Data signature time: " << std::chrono::duration_cast<std::chrono::microseconds>(total_sig_data_time).count() << " us" << std::endl;
  std::cout << "Query signature time: " << std::chrono::duration_cast<std::chrono::microseconds>(total_sig_query_time).count() << " us" << std::endl;
  std::cout << "Filter time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_filter_time).count() << " ms" << std::endl;
  std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() << " ms" << std::endl;

  std::cout << "------------- Overall Host Stats -------------" << std::endl;
  std::cout << "Setup Data time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("setup_data_start", "setup_data_end")).count()
            << " ms" << std::endl;
  std::cout << "Filter time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("filter_start", "filter_end")).count() << " ms"
            << std::endl;
  std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getOverallTime()).count() << " ms"
            << std::endl;

  if (args.inspect_candidates) {
    CandidatesInspector inspector;
    for (size_t i = 0; i < data_nodes; ++i) {
      auto count = candidates.getCandidatesCount(i);
      inspector.add(count);
      if (args.print_candidates) std::cerr << "Node " << i << ": " << count << std::endl;
    }
    inspector.finalize();
    std::cout << "------------- Filter Results -------------" << std::endl;
    std::cout << "# Total candidates: " << formatNumber(inspector.total) << std::endl;
    std::cout << "# Average candidates: " << formatNumber(inspector.avg) << std::endl;
    std::cout << "# Median candidates: " << formatNumber(inspector.median) << std::endl;
    std::cout << "# Zero candidates: " << formatNumber(inspector.zero_count) << std::endl;
  }

  sycl::free(tmp_buff, queue);
  sycl::free(query_signatures, queue);
  sycl::free(data_signatures, queue);
  sycl::free(candidates.candidates, queue);
  mbsm::destroyDeviceDataGraph(device_data_graph, queue);
  mbsm::destroyDeviceQueryGraph(device_query_graph, queue);
}