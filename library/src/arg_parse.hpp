#pragma once

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


class Args {
public:
  std::string fname = "/home/adecaro/subgraph-iso-soa/data/MBSM/pool.bin";
  bool print_candidates = false;
  int refinement_steps = 1;
  bool validate = false;
  bool query_data = false;
  std::string query_file;
  std::string data_file;
  size_t multiply_factor = 1;


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
    std::cout << "   -p: Print the number of candidates for each query node" << std::endl;
    std::cout << "   -i: Print the number of refined iterations. Default = 1" << std::endl;
    std::cout << "  -qd: Define the query file and the data file to read" << std::endl;
    std::cout << "   -m: Multiply the number of graphs by a factor. Default = 1" << std::endl;
  }

  void parseOption(std::string& arg, size_t& idx) {
    if (arg == "p") {
      print_candidates = true;
    } else if (arg == "i") {
      if (idx + 1 >= _argc) {
        printHelp();
        std::exit(1);
      }
      refinement_steps = std::stoi(_argv[++idx]);
    } else if (arg == "v") {
      validate = true;
    } else if (arg == "m") {
      if (idx + 1 >= _argc) {
        printHelp();
        std::exit(1);
      }
      multiply_factor = std::stoi(_argv[++idx]);
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