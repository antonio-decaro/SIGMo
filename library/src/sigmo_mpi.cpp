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
   MPI_Offset end = start + (max_lines * 2048);//(rank == nprocs - 1) ? file_size : start + chunk_size;
 
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

   lines.erase(std::remove_if(lines.begin(), lines.end(), [](const std::string &line) {
    return (std::count(line.begin(), line.end(), 'n') != 1 ||
            std::count(line.begin(), line.end(), 'e') != 1 ||
            std::count(line.begin(), line.end(), 'l') != 1);
    }), lines.end());
 
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
   TimeEvents host_time_events;
   
   size_t query_nodes = device_query_graph.total_nodes;
   size_t data_nodes = device_data_graph.total_nodes;
   
   size_t total_data_graphs = 0;
   MPI_Reduce(&num_data_graphs, &total_data_graphs, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

   if (mpi_rank == 0) {
     std::cout << "------------- Input Data -------------" << std::endl;
     std::cout << "Read data and query graphs" << std::endl;
     std::cout << "# Query Graphs " << num_query_graphs << std::endl;
     std::cout << "# Data Graphs " << total_data_graphs << std::endl;
     
     std::cout << "------------- Configs -------------" << std::endl;
     std::cout << "Filter domain: " << args.candidates_domain << std::endl;
     std::cout << "Filter Work Group Size: " << sigmo::device::deviceOptions.filter_work_group_size << std::endl;
     std::cout << "Join Work Group Size: " << sigmo::device::deviceOptions.join_work_group_size << std::endl;
     std::cout << "Find all: " << (args.find_all ? "Yes" : "No") << std::endl;
   }
   
   sigmo::candidates::Candidates candidates{queue, query_nodes, data_nodes};
   
   sigmo::signature::Signature<> signatures{queue, device_data_graph.total_nodes, device_query_graph.total_nodes};
   host_time_events.add("setup_data_start");
   host_time_events.add("setup_data_end");

  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    host_time_events.add("mpi_start");
  }
  // Synchronize all ranks to ensure they have completed their setup.
   auto e1 = signatures.generateDataSignatures(device_data_graph);
   e1.wait();
   
   auto e2 = signatures.generateQuerySignatures(device_query_graph);
   e2.wait();
   
   auto e3 = sigmo::isomorphism::filter::filterCandidates<sigmo::CandidatesDomain::Query>(queue, device_query_graph, device_data_graph, signatures, candidates);
   e3.wait();
   
   // Start refining candidate set.
   for (size_t ref_step = 1; ref_step <= args.refinement_steps; ++ref_step) {
     auto e1 = signatures.refineDataSignatures(device_data_graph, ref_step);
     e1.wait();
   
     auto e2 = signatures.refineQuerySignatures(device_query_graph, ref_step);
     e2.wait();
   
     auto e3 = sigmo::isomorphism::filter::refineCandidates<sigmo::CandidatesDomain::Query>(queue, device_query_graph, device_data_graph, signatures, candidates);
     e3.wait();
   }
   host_time_events.add("filter_end");
   
   std::chrono::duration<double> join_time{0};
   size_t* num_matches = sycl::malloc_shared<size_t>(1, queue);
   num_matches[0] = 0;
   if (!args.skip_join) {
     host_time_events.add("mapping_start");
     sigmo::isomorphism::mapping::GMCR gmcr{queue};
     gmcr.generateGMCR(device_query_graph, device_data_graph, candidates);
     host_time_events.add("mapping_end");
     host_time_events.add("join_start");
     auto join_e = sigmo::isomorphism::join::joinCandidates(queue, device_query_graph, device_data_graph, candidates, gmcr, num_matches, !args.find_all);
     join_e.wait();
     host_time_events.add("join_end");
   }
   MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_rank == 0) {
      host_time_events.add("mpi_end");
    }

  size_t total_matches = 0;
  MPI_Reduce(num_matches, &total_matches, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (mpi_rank == 0) {    
    std::cout << "------------- Results -------------" << std::endl;
    std::cout << "MPI time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("mpi_start", "mpi_end")).count() << " ms"
      << std::endl;
    std::cout << "# Total matches: " << formatNumber(total_matches) << std::endl;
    std::cout << "# Average matches: " << formatNumber(total_matches / mpi_size) << std::endl;
   }

   sycl::free(num_matches, queue);
   sigmo::destroyDeviceCSRGraph(device_data_graph, queue);
   sigmo::destroyDeviceCSRGraph(device_query_graph, queue);
   
   // Finalize MPI.
   MPI_Finalize();
   
   return 0;
 }