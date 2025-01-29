#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <bitset>

#include <msm.hpp>
#include "arg_parse.hpp"


void printGraph(msm::Graph& g, std::string tab = "\t") {
  for (int i = 0; i < g.getNumNodes(); i++) {
    std::cout << tab << "[" << i << "] " << g.getLabel(i) << ": ";
    for (auto& e : g.getNeighbours(i)) {
      std::cout << e << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char** argv) {
  Args args{argc, argv};

  sycl::queue q {sycl::gpu_selector_v};

  std::vector<msm::SubgraphIsomorphism> isos;
  std::vector<msm::Graph> data_graphs;
  std::vector<msm::Graph> query_graphs;

  for (auto data : args.data) {
    data_graphs.push_back(msm::utils::readGraphFromFile(data));
  }
  for (auto query : args.query) {
    query_graphs.push_back(msm::utils::readGraphFromFile(query));
  }
  for (auto& data : data_graphs) {
    isos.push_back(msm::SubgraphIsomorphism(data, {query_graphs}, q));
  }

  for (auto& iso : isos) {
    iso.run();
  }
  for (int i = 0; i < isos.size(); i++) {
    auto& iso = isos[i];
    std::cerr << "Waiting Instance " << i << std::endl;
    try {
      iso.wait_for(std::chrono::seconds(10));
    } catch (std::exception& e) {
      auto data = args.data[i / args.query.size()];
      auto query = args.query[i % args.query.size()];
      std::cerr << "[!] Time out for instance " << i << " | data: " << data << " | query: " << query << std::endl;
    }
  }

  for (int i = 0; i < args.data.size(); i++) {
    auto& iso = isos[i];
    std::cout << "Time [us]: " << iso.getExceutionTime() << std::endl;
    auto data = args.data[i].substr(args.data[i].find_last_of('/') + 1);
    std::cout << "Data: " << data << std::endl;
    for (int j = 0; j < args.query.size(); j++) {
      auto query = args.query[j].substr(args.query[j].find_last_of('/') + 1);
      std::cout << "Query: " << query << std::endl;
      auto ans = iso.getAnswers()[j];
      if (args.print_mapping) {
        if (ans.size() == 0) {
          std::cout << "No mappings found" << std::endl;
          continue;
        }
        for (auto& m : ans) {
          std::cout << "(" << m.first << " -> " << m.second << ") ";
        }
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }
  exit(0);
}