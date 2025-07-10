# SIGMo: Scalable Isomorphism Graph Matching on GPUs

SIGMo is a high-performance GPU framework for batched subgraph isomorphism, specifically designed for molecular matching tasks at scale. The framework enables efficient filtering and matching of molecules using GPUs across multiple vendors (NVIDIA, AMD, Intel) and supports both single-node and multi-node execution environments.

## Key Features

- Iterative vertex filtering strategy to reduce search space early.
- Support for multiple GPU backends using SYCL and native toolchains.
- Multi-GPU and multi-node scalability with MPI and SLURM.
- Includes state-of-the-art baselines for benchmarking: VF3, CuTS, and GSI.

## Repository Structure

```
├── library/             # SIGMo library
├── build/               # SIGMo build destination
├── benchmark/           # Scripts to run baseline frameworks (VF3, CuTS, GSI)
├── data/                # Input molecule data and generated formats
├── scripts/             # SLURM scripts and framework launchers
├── out/                 # Output results (auto-generated)
├── 1_build.sh           # Build script for frameworks and SIGMo
├── 2_init.sh            # Dataset conversion and initialization
├── 3_run.sh             # Unified runner for all experiments
├── 4_produce_plot.ipynb # Jupyter notebook to generate evaluation plots
```

## Quick Start

### Prerequisites

Install the following system packages and toolchains:
- Python ≥ 3.9
- CMake ≥ 3.10
- g++ 11.4.0 (different versions may lead to a fail during compilation)
- Intel oneAPI 2025.1.0
- CUDA toolkit (12.3), and ROCm (7.0.0) with Codeplay plug-in matching the oneAPI version (Optional if want to run tests on NVIDIA or AMD GPUs)
- NVIDIA DCGMI and NCU (Optional for NVIDIA metrics)
- `zstd` (Optional for ZINC dataset decompression)
- Intel MPI Library (Optional for multi-node support)

### 1. Compilation

First, load the environment (example for Intel oneAPI):

```bash
source /opt/intel/oneapi/setvars.sh # if oneAPI is installed globally on the machine
source ~/intel/oneapi/setvars.sh # if oneAPI is installed locally in the user home directory
```

Then build SIGMo and/or other frameworks:

```bash
./1_build.sh -b=SIGMO \
  --sigmo-arch=nvidia_gpu_sm_80 \ # here goes the GPU architecture (NVIDIA A100)
  --sigmo-compiler=$(which icpx)
```

For other GPUs:

- `nvidia_gpu_sm_70` for V100
- `amd_gpu_gfx908` for MI100
- `intel_gpu_pvc` for Max 1100

To include VF3, CuTS, GSI:

```bash
./1_build.sh -b=VF3,CuTS,GSI,SIGMO ...
```
> __Note__: `libreadline-dev` is required to compile GSI. If not installed, it will be automatically downloaded and compiled locally.

### 2. Dataset Preparation

```bash
./2_init.sh -b=VF3,CuTS,GSI,SIGMO
```

To download the dataset used for multi-node scalability tests:

```bash
./2_init.sh -b=SIGMO --zinc=/path/where/zinc/will/be/downloaded # ~6GB of compressed dataset
```
> __Note__: This script uses `zstd` to decompress the downloaded dataset.

### 3. Running Experiments

#### Run SOTA frameworks:

```bash
./3_run.sh -b=VF3,CuTS,GSI
```

#### Run SIGMo only with all configurations:

```bash
./3_run.sh -b=SIGMO \
  -e=core,diameter,dataset-scale,gpu-metrics
```
This command launches all single-node experiments for SIGMo:
- `core` Evaluates the SIGMo's filtering strategy and its performance.
- `diameter` Groups query graphs by diameter and runs experiments per group;
- `dataset-scale` Assesses single-node scalability by varying dataset size;
- `gpu-metrics` Collects GPU performance metrics (requires NVIDIA NCU and DCGMI tools).

You can also run a single experiment or a subset by sepcifying the desired components with `-e`. For example: `-e=core,diameter`.

#### Run portability experiments:
On each hardware execute the following command.
```bash
./3_run.sh -b=SIGMO -e=core
# Rename the output file:
mv ./out/SIGMO/sigmo_results.csv ./out/SIGMO/sigmo_results_<vendor>.csv
```
> Upon completion of all experiments, the renamed result files should be gathered and transferred to a single machine designated for artifact analysis. All files must be placed in the directory ./out/SIGMO/ on that machine.

#### Run multi-node scalability with SLURM:

1. Edit `./scripts/slurm/run_slurm.sh` with your SLURM account and partition.
2. Run:

```bash
./3_run.sh -b=SIGMO -e=mpi --zinc=/path/to/zinc
```

### Plotting Results

Use the Jupyter notebook to generate all evaluation plots.
Run the notebook cells starting from the "Initialization" section. Each section generates plots used for evaluation and benchmarking.

---

## License

Apache 2 License

## Acknowledgements

This project leverages open source tools and benchmarks including VF3, CuTS, and GSI.

---

## Contact

For issues or collaboration requests, please open an issue or contact the maintainer directly.

