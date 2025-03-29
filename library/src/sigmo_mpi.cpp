/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

 #include <mpi.h>
 #include <sycl/sycl.hpp>
 #include "./utils.hpp"
 #include <numeric>
 #include <sigmo.hpp>
 #include <sstream>
 #include <vector>
 #include <string>
 #include <iostream>
 
 // Helper function to read a text file using MPI-IO and split it into lines.
 // Each MPI rank reads its file chunk, adjusts for incomplete lines at the boundaries,
 // and returns at most max_lines complete lines.
 std::vector<std::string> loadFileLinesMPI(const std::string &filename, int max_lines) {
   MPI_File mpi_file;
   // Open the file collectively for reading.
   MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);
 
   // Get the size of the file in bytes.
   MPI_Offset file_size;
   MPI_File_get_size(mpi_file, &file_size);
 
   // Get the MPI rank and number of processes.
   int rank, nprocs;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
 
   // Compute the basic chunk size (integer division).
   MPI_Offset chunk_size = file_size / nprocs;
   MPI_Offset start = rank * chunk_size;
   MPI_Offset end = (rank == nprocs - 1) ? file_size : start + chunk_size;
 
   // Read extra bytes to cover a potential partial line at the chunk boundary.
   const int extra = 1024;
   std::vector<char> buffer((end - start) + extra, '\0');
 
   // Read the file chunk collectively.
   MPI_File_read_at_all(mpi_file, start, buffer.data(), buffer.size(), MPI_CHAR, MPI_STATUS_IGNORE);
   MPI_File_close(&mpi_file);
 
   // Convert the buffer into a string.
   std::string data(buffer.data(), buffer.size());
 
   // For non-first ranks, skip a potential partial first line.
   if (rank != 0) {
     size_t first_newline = data.find('\n');
     if (first_newline != std::string::npos) {
       data = data.substr(first_newline + 1);
     }
   }
 
   // For non-last ranks, remove any partial last line by cutting at the final newline.
   if (rank != nprocs - 1) {
     size_t last_newline = data.rfind('\n');
     if (last_newline != std::string::npos) {
       data = data.substr(0, last_newline);
     }
   }
 
   // Split the data into individual lines.
   std::istringstream iss(data);
   std::string line;
   std::vector<std::string> lines;
   while (std::getline(iss, line) && lines.size() < static_cast<size_t>(max_lines)) {
     lines.push_back(line);
   }
 
   return lines;
 }
 
 int main(int argc, char** argv) {
   // Initialize MPI.
   MPI_Init(&argc, &argv);
   
   // Get MPI rank and size (useful for debugging or further logic).
   int mpi_rank, mpi_size;
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
 
   // Initialize program arguments and device options.
   Args args{argc, argv, sigmo::device::deviceOptions};
 
   // Set up a SYCL queue with GPU selection and profiling enabled.
   sycl::queue queue{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
   size_t gpu_mem = queue.get_device().get_info<sycl::info::device::global_mem_size>();
   std::string gpu_name = queue.get_device().get_info<sycl::info::device::name>();
 
   // --- Read data_file in parallel using MPI-IO ---
   // Set the maximum number of lines per node.
   std::vector<std::string> data_lines = loadFileLinesMPI(args.data_file, args.max_data_graphs);
   
   // Debug print: show how many lines this MPI rank read.
   std::cout << "MPI Rank " << mpi_rank << " read " << data_lines.size()
             << " lines from " << args.data_file << std::endl;
   
   // TODO: Parse the read lines into CSR graphs.
   // You need to implement sigmo::io::parseCSRGraphsFromLines to convert the lines into the expected CSR graphs.
   auto data_graphs = sigmo::io::loadCSRGraphsFromLines(data_lines);
   
   // Load query graphs using the existing (serial) routine.
   auto query_graphs = sigmo::io::loadCSRGraphsFromFile(args.query_file);
   
   size_t num_query_graphs = query_graphs.size();
   size_t num_data_graphs = data_graphs.size();
   
   sigmo::DeviceBatchedCSRGraph device_data_graph;
   sigmo::DeviceBatchedCSRGraph device_query_graph;
   
   device_query_graph = sigmo::createDeviceCSRGraph(queue, query_graphs);
   device_data_graph = sigmo::createDeviceCSRGraph(queue, data_graphs);
   
   size_t data_graph_bytes = sigmo::getDeviceGraphAllocSize(device_data_graph);
   size_t query_graphs_bytes = sigmo::getDeviceGraphAllocSize(device_query_graph);
   
   std::vector<std::chrono::duration<double>> data_sig_times, query_sig_times, filter_times;
   
   size_t query_nodes = device_query_graph.total_nodes;
   size_t data_nodes = device_data_graph.total_nodes;
   
   std::cout << "------------- Input Data -------------" << std::endl;
   std::cout << "Read data and query graphs" << std::endl;
   std::cout << "# Query Nodes " << query_nodes << std::endl;
   std::cout << "# Query Graphs " << num_query_graphs << std::endl;
   std::cout << "# Data Nodes " << data_nodes << std::endl;
   std::cout << "# Data Graphs " << num_data_graphs << std::endl;
   
   std::cout << "------------- Configs -------------" << std::endl;
   std::cout << "Filter domain: " << args.candidates_domain << std::endl;
   std::cout << "Filter Work Group Size: " << sigmo::device::deviceOptions.filter_work_group_size << std::endl;
   std::cout << "Join Work Group Size: " << sigmo::device::deviceOptions.join_work_group_size << std::endl;
   std::cout << "Find all: " << (args.find_all ? "Yes" : "No") << std::endl;
   
   TimeEvents host_time_events;
   
   std::cout << "------------- Setup Data -------------" << std::endl;
   std::cout << "Allocated " << getBytesSize(data_graph_bytes) << " for graph data" << std::endl;
   std::cout << "Allocated " << getBytesSize(query_graphs_bytes) << " for query data" << std::endl;
   
   size_t source_nodes = args.isCandidateDomainData() ? data_nodes : query_nodes;
   size_t target_nodes = args.isCandidateDomainData() ? query_nodes : data_nodes;
   sigmo::candidates::Candidates candidates{queue, source_nodes, target_nodes};
   size_t candidates_bytes = candidates.getAllocationSize();
   std::cout << "Allocated " << getBytesSize(candidates_bytes) << " for candidates" << std::endl;
   
   sigmo::signature::Signature<> signatures{queue, device_data_graph.total_nodes, device_query_graph.total_nodes};
   size_t data_signatures_bytes = signatures.getDataSignatureAllocationSize();
   std::cout << "Allocated " << getBytesSize(data_signatures_bytes) << " for data signatures" << std::endl;
   size_t query_signatures_bytes = signatures.getQuerySignatureAllocationSize();
   std::cout << "Allocated " << getBytesSize(query_signatures_bytes) << " for query signatures" << std::endl;
   size_t tmp_buff_bytes = std::max(data_signatures_bytes, query_signatures_bytes);
   std::cout << "Allocated " << getBytesSize(tmp_buff_bytes) << " for temporary buffer" << std::endl;
   host_time_events.add("setup_data_start");
   host_time_events.add("setup_data_end");
   
   std::cout << "Total allocated memory: "
             << getBytesSize(
                    data_signatures_bytes + query_signatures_bytes + candidates_bytes + tmp_buff_bytes + data_graph_bytes + query_graphs_bytes, false)
             << " out of " << getBytesSize(gpu_mem) << " available on " << gpu_name << std::endl;
   


  if (mpi_rank == 0) {
    host_time_events.add("mpi_start");
  }
  // Synchronize all ranks to ensure they have completed their setup.
  MPI_Barrier(MPI_COMM_WORLD);

   std::cout << "------------- Runtime Filter Phase -------------" << std::endl;
   host_time_events.add("filter_start");
   std::cout << "[*] Initialization Step:" << std::endl;
   std::chrono::duration<double> time;
   auto e1 = signatures.generateDataSignatures(device_data_graph);
   queue.wait_and_throw();
   time = e1.getProfilingInfo();
   data_sig_times.push_back(time);
   std::cout << "- Data signatures generated in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;
   
   auto e2 = signatures.generateQuerySignatures(device_query_graph);
   queue.wait_and_throw();
   time = e2.getProfilingInfo();
   query_sig_times.push_back(time);
   std::cout << "- Query signatures generated in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;
   
   auto e3 = sigmo::isomorphism::filter::filterCandidates<sigmo::CandidatesDomain::Query>(queue, device_query_graph, device_data_graph, signatures, candidates);
   queue.wait_and_throw();
   time = e3.getProfilingInfo();
   filter_times.push_back(time);
   std::cout << "- Candidates filtered in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;
   
   // Start refining candidate set.
   for (size_t ref_step = 1; ref_step <= args.refinement_steps; ++ref_step) {
     std::cout << "[*] Refinement step " << ref_step << ":" << std::endl;
   
     auto e1 = signatures.refineDataSignatures(device_data_graph, ref_step);
     queue.wait_and_throw();
     time = e1.getProfilingInfo();
     data_sig_times.push_back(time);
     std::cout << "- Data signatures refined in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;
   
     auto e2 = signatures.refineQuerySignatures(device_query_graph, ref_step);
     queue.wait_and_throw();
     time = e2.getProfilingInfo();
     query_sig_times.push_back(time);
     std::cout << "- Query signatures refined in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;
   
     auto e3 = sigmo::isomorphism::filter::refineCandidates<sigmo::CandidatesDomain::Query>(queue, device_query_graph, device_data_graph, signatures, candidates);
     queue.wait_and_throw();
     time = e3.getProfilingInfo();
     filter_times.push_back(time);
     std::cout << "- Candidates refined in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;
   }
   host_time_events.add("filter_end");
   
   std::chrono::duration<double> join_time{0};
   size_t* num_matches = sycl::malloc_shared<size_t>(1, queue);
   num_matches[0] = 0;
   if (!args.skip_join) {
     std::cout << "[*] Generating DQCR" << std::endl;
     host_time_events.add("mapping_start");
     sigmo::isomorphism::mapping::GMCR gmcr{queue};
     gmcr.generateGMCR(device_query_graph, device_data_graph, candidates);
     host_time_events.add("mapping_end");
     std::cout << "[*] Starting Join" << std::endl;
     host_time_events.add("join_start");
     auto join_e = sigmo::isomorphism::join::joinCandidates(queue, device_query_graph, device_data_graph, candidates, gmcr, num_matches, !args.find_all);
     join_e.wait();
     join_time = join_e.getProfilingInfo();
     host_time_events.add("join_end");
   }
   std::cout << "[!] End" << std::endl;
   MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_rank == 0) {
      host_time_events.add("mpi_end");
    }
   
   std::cout << "------------- Overall GPU Stats -------------" << std::endl;
   std::chrono::duration<double> total_sig_query_time =
       std::accumulate(query_sig_times.begin(), query_sig_times.end(), std::chrono::duration<double>(0));
   std::chrono::duration<double> total_sig_data_time = std::accumulate(data_sig_times.begin(), data_sig_times.end(), std::chrono::duration<double>(0));
   std::chrono::duration<double> total_filter_time = std::accumulate(filter_times.begin(), filter_times.end(), std::chrono::duration<double>(0));
   std::chrono::duration<double> total_time = total_sig_data_time + total_filter_time + total_sig_query_time + join_time;
   std::cout << "Data signature GPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_sig_data_time).count() << " ms" << std::endl;
   std::cout << "Query signature GPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_sig_query_time).count() << " ms" << std::endl;
   std::cout << "Filter time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_filter_time).count() << " ms" << std::endl;
   if (args.skip_join) {
     std::cout << "Join GPU time: skipped" << std::endl;
   } else {
     std::cout << "Join GPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(join_time).count() << " ms" << std::endl;
   }
   std::cout << "Total GPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() << " ms" << std::endl;
   
   std::cout << "------------- Overall Host Stats -------------" << std::endl;
   std::cout << "Setup Data time: "
             << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("setup_data_start", "setup_data_end")).count()
             << " ms (not included in total)" << std::endl;
   std::cout << "Filter time: "
             << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("filter_start", "filter_end")).count() << " ms"
             << std::endl;
   if (args.skip_join) {
     std::cout << "Mapping time: skipped" << std::endl;
     std::cout << "Join time: skipped" << std::endl;
   } else {
     std::cout << "Mapping time: "
               << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("mapping_start", "mapping_end")).count() << " ms"
               << std::endl;
     std::cout << "Join time: "
               << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("join_start", "join_end")).count() << " ms"
               << std::endl;
   }
   std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getTimeFrom("setup_data_end")).count()
             << " ms" << std::endl;
  if (mpi_rank == 0) {
     std::cout << "MPI time: "
               << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("mpi_start", "mpi_end")).count() << " ms"
               << std::endl;
   }
   
   CandidatesInspector inspector;
   auto host_candidates = candidates.getHostCandidates();
   for (size_t i = 0; i < (args.isCandidateDomainData() ? data_nodes : query_nodes); ++i) {
     auto count = host_candidates.getCandidatesCount(i);
     inspector.add(count);
     if (args.print_candidates)
       std::cerr << "Node " << i << ": " << count << std::endl;
   }
   inspector.finalize();
   std::cout << "------------- Results -------------" << std::endl;
   std::cout << "# Total candidates: " << formatNumber(inspector.total) << std::endl;
   std::cout << "# Average candidates: " << formatNumber(inspector.avg) << std::endl;
   std::cout << "# Median candidates: " << formatNumber(inspector.median) << std::endl;
   std::cout << "# Zero candidates: " << formatNumber(inspector.zero_count) << std::endl;
   if (!args.skip_join) {
     std::cout << "# Matches: " << formatNumber(num_matches[0]) << std::endl;
   }
   
   sycl::free(num_matches, queue);
   sigmo::destroyDeviceCSRGraph(device_data_graph, queue);
   sigmo::destroyDeviceCSRGraph(device_query_graph, queue);
   
   // Finalize MPI.
   MPI_Finalize();
   
   return 0;
 }