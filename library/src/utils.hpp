/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <algorithm>
#include <cmath>
#include <cxxopts.hpp> // Include cxxopts header
#include <filesystem>
#include <iostream>
#include <sigmo.hpp>
#include <stdexcept>
#include <string>
#include <vector>

class Args {
public:
  struct Filter {
    bool active = false;
    size_t max_nodes = 0;
    size_t min_nodes = static_cast<size_t>(-1);
  };

  bool print_candidates = false;
  bool skip_join = false;
  int refinement_steps = 0;
  bool query_data = false;
  std::string query_file;
  std::string data_file;
  size_t multiply_factor_query = 1;
  size_t multiply_factor_data = 1;
  bool find_all = false;
  std::string candidates_domain = "query";
  size_t join_work_group_size = 0;
  size_t max_data_graphs = 1000000;
  size_t max_query_graphs = 1000;
  Args::Filter query_filter;
  bool skip_print_candidates = false;

  Args(int& argc, char**& argv, sigmo::device::DeviceOptions& device_options) {
    cxxopts::Options options(argv[0], "Command line options");
    options.add_options()("p,print-candidates", "Print the number of candidates for each query node", cxxopts::value<bool>(print_candidates))(
        "i,iterations", "Number of refinement iterations", cxxopts::value<int>(refinement_steps))(
        "Q", "Define the query file to read", cxxopts::value<std::string>(query_file))(
        "D", "Define the data file to read", cxxopts::value<std::string>(data_file))(
        "c,candidates-domain", "Select the candidates domain [query, data]", cxxopts::value<std::string>(candidates_domain))(
        "m,multiply", "Multiply the number of all graphs by a factor", cxxopts::value<size_t>())(
        "d,mul-data", "Multiply the number of data graphs by a factor", cxxopts::value<size_t>(multiply_factor_data))(
        "q,mul-query", "Multiply the number of query graphs by a factor", cxxopts::value<size_t>(multiply_factor_query))(
        "skip-join", "Skip the join phase", cxxopts::value<bool>(skip_join))("h,help", "Print usage")(
        "find-all", "Find all matches without stopping at the first one", cxxopts::value<bool>(find_all))(
        "query-filter", "Apply a filter to the query graphs. Format: min[:max]", cxxopts::value<std::string>())(
        "skip-candidates-analysis", "Skip the analysis of the candidates", cxxopts::value<bool>(skip_print_candidates))(
        "max-data-graphs", "Limit the number of data graphs", cxxopts::value<size_t>(max_data_graphs))(
        "max-query-graphs", "Limit the number of query graphs", cxxopts::value<size_t>(max_query_graphs))(
        "join-work-group", "Set the work group size for the join kernel. Default 128.", cxxopts::value<size_t>(device_options.join_work_group_size))(
        "filter-work-group",
        "Set the work group size for the filter kernel. Default 512.",
        cxxopts::value<size_t>(device_options.filter_work_group_size));
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      std::exit(0);
    }

    if (result.count("Q") && result.count("D")) {
      query_data = true;
    } else if (result.count("Q") || result.count("D")) {
      throw std::runtime_error("Both query and data files must be provided");
    }

    if (result.count("multiply")) { multiply_factor_data = multiply_factor_query = result["multiply"].as<size_t>(); }

    if (result.count("query-filter")) {
      query_filter.active = true;
      std::string filter_arg = result["query-filter"].as<std::string>();
      size_t colon_pos = filter_arg.find(':');
      if (colon_pos != std::string::npos) {
        query_filter.min_nodes = std::stoi(filter_arg.substr(0, colon_pos));
        query_filter.max_nodes = std::stoi(filter_arg.substr(colon_pos + 1));
      } else {
        query_filter.min_nodes = std::stoi(filter_arg);
      }
    }
  }

  bool isCandidateDomainQuery() const { return candidates_domain == "query"; }
  bool isCandidateDomainData() const { return candidates_domain == "data"; }
};

struct TimeEvents {
  std::vector<std::pair<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>>> events;

  void add(std::string name) {
    auto now = std::chrono::high_resolution_clock::now();
    this->events.emplace_back(name, now);
  }

  std::chrono::duration<double> getOverallTime() { return this->events.back().second - this->events.front().second; }
  std::chrono::duration<double> getTimeTill(std::string event) {
    auto it = std::find_if(this->events.begin(), this->events.end(), [event](auto& e) { return e.first == event; });
    if (it == this->events.end()) throw std::runtime_error("Event not found");
    return it->second - this->events.front().second;
  }
  std::chrono::duration<double> getTimeFrom(std::string event) {
    auto it = std::find_if(this->events.begin(), this->events.end(), [event](auto& e) { return e.first == event; });
    if (it == this->events.end()) throw std::runtime_error("Event not found");
    return this->events.back().second - it->second;
  }
  std::chrono::duration<double> getEventTime(std::string event) {
    auto it = std::find_if(this->events.begin(), this->events.end(), [event](auto& e) { return e.first == event; });
    if (it == this->events.end()) throw std::runtime_error("Event not found");
    return it->second - (it - 1)->second;
  }
  std::chrono::duration<double> getRangeTime(std::string first_event, std::string last_event) {
    auto first = std::find_if(this->events.begin(), this->events.end(), [first_event](auto& e) { return e.first == first_event; });
    auto last = std::find_if(this->events.begin(), this->events.end(), [last_event](auto& e) { return e.first == last_event; });
    if (first == this->events.end() || last == this->events.end()) throw std::runtime_error("Event not found");
    // check also if first is before last
    if (first - last > 0) throw std::runtime_error("First event is after last event");
    return last->second - first->second;
  }
  void clear() { this->events.clear(); }
};
struct DotSeparated : std::numpunct<char> {
protected:
  char do_thousands_sep() const override { return '.'; }
  std::string do_grouping() const override { return "\3"; }
};

std::string formatNumber(size_t number) {
  std::stringstream ss;
  ss.imbue(std::locale(std::locale(), new DotSeparated));
  ss << std::fixed << number;
  return ss.str();
}

std::string getBytesSize(size_t num_bytes, bool round = true) {
  // convert to the largest unit
  double bytes = static_cast<double>(num_bytes);
  std::string unit = "B";
  if (num_bytes >= 1024 * 1024 * 1024) {
    bytes /= 1024 * 1024 * 1024;
    unit = "GB";
  } else if (num_bytes >= 1024 * 1024) {
    bytes /= 1024 * 1024;
    unit = "MB";
  } else if (num_bytes >= 1024) {
    bytes /= 1024;
    unit = "KB";
  }

  if (round) return std::to_string(static_cast<int>(std::round(bytes))) + " " + unit;
  return std::to_string(bytes) + " " + unit;
}

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