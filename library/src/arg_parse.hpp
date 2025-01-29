#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>

typedef struct Args {
  std::string exec_name;
  std::vector<std::string> data;
  std::vector<std::string> query;
  bool print_mapping {false};


  Args(int argc, char** argv) {
    exec_name = std::string(argv[0]);
    if (argc < 3) {
      help();
      exit(1);
    }

    std::string data_fname = std::string(argv[1]);
    if (std::filesystem::is_directory(data_fname)) {
      for (auto& p : std::filesystem::directory_iterator(data_fname)) {
        data.push_back(p.path().string());
      }
    } else {
      data.push_back(data_fname);
    }
    std::string query_fname = std::string(argv[2]);
    if (std::filesystem::is_directory(query_fname)) {
      for (auto& p : std::filesystem::directory_iterator(query_fname)) {
        query.push_back(p.path().string());
      }
    } else {
      query.push_back(query_fname);
    }

    for (int i = 3; i < argc; i++) {
      std::string arg = std::string(argv[i]);
      if (arg == "-h") {
        help();
        exit(0);
      } else if (arg == "-p") {
        print_mapping = true;
      } else {
        std::cout << "Unknown argument: " << arg << std::endl;
        help();
        exit(1);
      }
    }
  }

  void help() {
    std::cout << ("Usage: " + exec_name + " <data file or directory> <query file or directory>") << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "\t-p\tPrint mapping" << std::endl;
  }

} Args;