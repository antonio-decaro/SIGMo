# Subgraph Isomorphism Benchmarks
Testing the state of the art for subgraph isomorphism.

## Repository
The repository is structured as follows:
- `repo` this folder contains all the other repository of the algorithms to test;
- `scripts` contains all the script to parse data in the several formats required by all the algorithms;
- `data` contains all the query and data graphs to be tested;

## Repeating Experiments
### 1. Init Script
First of all launch the `init.sh` script.
Optional arguments are:
- `-b=bench1,bench2` the list of benchmarks. If it's left blank, all the benchmark will be evaluated;
- `--data-limit=x` the number of data graph to process;
- `--query-limit=x` the number of query graph to process;
