# Subgraph Isomorphism Benchmarks
Testing the state of the art for subgraph isomorphism.

## Requirements

## Repository
The repository is structured as follows:
- `repo` this folder contains all the other repository of the algorithms to test;
- `scripts` contains all the script to parse data in the several formats required by all the algorithms;
- `data` contains all the query and data graphs to be tested;

## Repeating Experiments
### 1. Init Repository
Init the repository by launching:
```bash
./download.sh
```
This will initialize the repository with all the dependencies.

### 2. Data Script
Second, launch the `init.sh` script to initialize data.
Optional arguments are:
- `-b=bench1,bench2` the list of benchmarks. If it's left blank, all the benchmark will be evaluated;
- `--data-limit=x` the number of data graph to process;
- `--query-limit=x` the number of query graph to process;
