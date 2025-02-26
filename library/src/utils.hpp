#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


class Args {
public:
  std::string fname = "/home/adecaro/subgraph-iso-soa/data/MBSM/pool.bin";
  bool print_candidates = false;
  bool inspect_candidates = false;
  int refinement_steps = 0;
  bool query_data = false;
  std::string query_file;
  std::string data_file;
  size_t multiply_factor_query = 1;
  size_t multiply_factor_data = 1;


  Args(int& argc, char**& argv) : _argc(argc), _argv(argv) {
    for (size_t i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg[0] == '-') {
        arg = arg.substr(1);
        parseOption(arg, i);
      } else {
        fname = argv[i];
      }
    }
  }

private:
  int& _argc;
  char**& _argv;
  void printHelp() {
    std::cout << "Usage: " << this->_argv[0] << " [options] [pool.bin]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "   -v: Inspect the candidates" << std::endl;
    std::cout << "   -p: Print the number of candidates for each query node" << std::endl;
    std::cout << "   -i: Print the number of refined iterations. Default = 1" << std::endl;
    std::cout << "  -qd: Define the query file and the data file to read" << std::endl;
    std::cout << "   -m: Multiply the number of all graphs by a factor. Default = 1" << std::endl;
    std::cout << "  -mq: Multiply the number of query graphs by a factor. Default = 1" << std::endl;
    std::cout << "  -md: Multiply the number of data graphs by a factor. Default = 1" << std::endl;
  }

  void parseOption(std::string& arg, size_t& idx) {
    if (arg == "p") {
      print_candidates = true;
      print_candidates = true;
    } else if (arg == "i") {
      if (idx + 1 >= _argc) {
        printHelp();
        std::exit(1);
      }
      refinement_steps = std::stoi(_argv[++idx]);
    } else if (arg == "v") {
      inspect_candidates = true;
    } else if (arg == "p") {
      print_candidates = true;
      inspect_candidates = true;
    } else if (arg == "m") {
      if (idx + 1 >= _argc) {
        printHelp();
        std::exit(1);
      }
      multiply_factor_data = multiply_factor_query = std::stoi(_argv[++idx]);
    } else if (arg == "md") {
      if (idx + 1 >= _argc) {
        printHelp();
        std::exit(1);
      }
      multiply_factor_data = std::stoi(_argv[++idx]);
    } else if (arg == "mq") {
      if (idx + 1 >= _argc) {
        printHelp();
        std::exit(1);
      }
      multiply_factor_query = std::stoi(_argv[++idx]);
    } else if (arg == "qd") {
      if (idx + 2 >= _argc) {
        printHelp();
        std::exit(1);
      }
      query_data = true;
      query_file = _argv[++idx];
      data_file = _argv[++idx];
    } else {
      printHelp();
      std::exit(1);
    }
  }
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