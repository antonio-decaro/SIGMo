# SIGMo: Scalable Isomorphism Graph Matching on GPUs

SIGMo is a high-performance GPU framework for batched subgraph isomorphism, specifically designed for molecular matching tasks at scale. The framework enables efficient filtering and matching of molecules using GPUs across multiple vendors (NVIDIA, AMD, Intel) and supports both single-node and multi-node execution environments.

## Key Features

- Iterative vertex filtering strategy to reduce search space early.
- Support for multiple GPU backends using SYCL and native toolchains.
- Multi-GPU and multi-node scalability with MPI and SLURM.
- Includes state-of-the-art baselines for benchmarking: VF3, CuTS, and GSI.

## Repository Structure

```
├── benchmarks/          # Scripts to run baseline frameworks (VF3, CuTS, GSI)
├── build/               # SIGMo build destination (directory is auto-generated)
├── data/                # Input molecule data and generated formats
├── library/             # SIGMo library
├── out/                 # Output results (auto-generated)
├── scripts/             # Framework launchers, SLURM scripts, and utility scripts
├── 1_build.sh           # Build script for frameworks and SIGMo
├── 2_init.sh            # Dataset conversion and initialization
├── 3_run.sh             # Unified runner for all experiments
├── 4_produce_plot.ipynb # Jupyter notebook to generate evaluation plots
```

## Quick Start

### Requirements

The following core software is required to build and run SIGMo:
- [Intel oneAPI Base Toolkit 2025.1.0](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux) – you can use any installer method, e.g., offline/online installer, `apt` or `yum` package manager
- CUDA toolkit (12.3) with [Codeplay plug-in](https://developer.codeplay.com/products/oneapi/nvidia/download) matching the oneAPI version if running tests on NVIDIA GPU – required for SIGMo assessment 
- ROCm (7.0.0) with [Codeplay plug-in](https://developer.codeplay.com/products/oneapi/amd/download) matching the oneAPI version if running tests on AMD GPU
- Python ≥ 3.9 – Packages are defined in `scripts/requirements.txt` and will be installed automatically
- CMake ≥ 3.10
- g++ 11.4.0 – different versions may lead to a fail during compilation
- `git` – for cloning repositories and submodules
- [NVIDIA DCGM](https://developer.nvidia.com/dcgm) and [NVIDA NCU](https://developer.nvidia.com/nsight-compute) – optional, but required for GPU metrics collection

For multi-node experiments, the following additional software is required:
- [SLURM](https://slurm.schedmd.com/) – for job scheduling
- [`zstd`](https://github.com/facebook/zstd) – for ZINC dataset decompression
- Intel MPI Library 2021.11 – it comes with Intel oneAPI Base Toolkit

#### Installing oneAPI
You can install Intel oneAPI Base Toolkit using the [official guide](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux). The recommended way is to use the online installer, which will download the required components during installation. Alternatively, you can use the offline installer if you have a stable internet connection.
After downloading the installer, run it with the following command:

```bash
sudo sh ./intel-oneapi-base-toolkit-2025.2.0.592.sh -a --silent --eula accept
```
This will install the oneAPI Base Toolkit in the default location (`/opt/intel/oneapi`).
Alternatively, you can install it in your home directory by running the installer without `sudo`.

After downloading and installing the oneAPI Base Toolkit, on systems equipped with NVIDIA or AMD GPUs, you will also need to install the Codeplay plug-in to enable SYCL support. You can follow the instructions on the [Codeplay website](https://developer.codeplay.com/products/oneapi) to install the plug-in OR use the following commands:

For NVIDIA GPUs:
```bash
curl -LOJ "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=nvidia&filters[]=2025.1.0&filters[]=linux"
chmod +x neapi-for-nvidia-gpus-2025.1.0-rocm-all-linux.sh
sudo ./oneapi-for-nvidia-gpus-2025.1.0-rocm-all-linux.sh
```
For AMD GPUs:
```bash
curl -LOJ "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=amd&filters[]=2025.1.0&filters[]=linux"
chmod +x oneapi-for-amd-gpus-2025.1.0-rocm-6.3-linux.sh
sudo ./oneapi-for-amd-gpus-2025.1.0-rocm-6.3-linux.sh
```
In both installations, avoid `sudo` if you installed oneAPI in your home directory.
> __Note__: The Codeplay plug-in is required for SIGMo to work with NVIDIA and AMD GPUs. If you are using Intel GPUs, you can skip this step. The plug-in version must match the oneAPI version you have installed.

### 1. Compilation

First, load the environment (example for Intel oneAPI):

```bash
source /opt/intel/oneapi/setvars.sh # if oneAPI is installed globally on the machine
source ~/intel/oneapi/setvars.sh # if oneAPI is installed locally in the user home directory
```

Then build SIGMo and/or other frameworks:

```bash
./1_build.sh -b=SIGMO \
  --sigmo-arch=nvidia_gpu_sm_80 # here goes the GPU architecture (NVIDIA A100)
```

For other GPUs:

- `nvidia_gpu_sm_70` for NVIDIA V100 (and V100S)
- `amd_gpu_gfx908` for AMD MI100
- `intel_gpu_pvc` for Intel Max 1100
- the list of supported architectures can be found in the [Intel LLVM User Manual](https://github.com/intel/llvm/blob/sycl/sycl/doc/UsersManual.md)

To include VF3, CuTS, GSI:

```bash
./1_build.sh -b=VF3,CuTS,GSI,SIGMO ...
```
> __Note__: `libreadline-dev` is required to compile GSI. If not installed, it will be automatically downloaded and compiled locally.
Additionally, make sure to have `g++` version 11.4.0 installed, as other versions may lead to compilation errors.

---

### 2. Dataset Preparation

```bash
./2_init.sh -b=VF3,CuTS,GSI,SIGMO
```

To download the dataset used for multi-node scalability tests:

```bash
./2_init.sh -b=SIGMO --zinc=/path/where/zinc/will/be/downloaded # ~6GB of compressed dataset
```
> __Note__: This script uses `zstd` to decompress the downloaded dataset.

---

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
- `gpu-metrics` Collects GPU performance metrics (requires NVIDIA NCU and DCGM tools).

You can also run a single experiment or a subset by sepcifying the desired components with `-e`. For example: `-e=core,diameter`.

#### Run portability experiments:
On each hardware execute the following command.
```bash
./3_run.sh -b=SIGMO -e=core
# Rename the output file:
mv ./out/SIGMO/sigmo_results.csv ./out/SIGMO/sigmo_results_<vendor>.csv
```
Where `<vendor>` is the GPU vendor (e.g., `nvidia`, `amd`, `intel`).
> Upon completion of all experiments, the renamed result files should be gathered and transferred to a single machine designated for artifact analysis. All files must be placed in the directory ./out/SIGMO/ on that machine.

#### Run multi-node scalability with SLURM:

1. Edit `./scripts/slurm/run_slurm.sh` with your SLURM account and partition.
2. Run:

```bash
./3_run.sh -b=SIGMO -e=mpi --zinc=/path/where/zinc/has/been/downloaded
```

---

### 4. Plotting Results

Use the Jupyter notebook to generate all evaluation plots. We reccomend using VSCode to run the notebook, as it provides a convenient interface for executing cells and visualizing results.
Run the notebook cells starting from the "Initialization" section. Each section generates plots used for evaluation and benchmarking.

If you prefer to run the notebook locally, follow these steps:
1. From the root directory, run: `source .venv/bin/activate` to activate the virtual environment (If you followed the previous instructions, this should be already set up. Otherwise, you can create a virtual environment with `python -m venv .venv`, activate it with `source .venv/bin/activate`, and run `pip install -r scripts/requirements.txt` to install the required packages)
2. Install the notebook package: `pip install notebook`
3. Start the Jupyter server: `jupyter notebook`
4. Open the web browser and navigate to the notebook `4_produce_plot.ipynb`
5. Run the notebook cells starting from the "Initialization" section

---

## License
This project is licensed under the [Apache 2 License](./LICENSE)

## Acknowledgements

This project leverages open source tools and benchmarks including [VF3](https://github.com/MiviaLab/vf3lib), [CuTS](https://dl.acm.org/doi/10.1145/3458817.3476214), and [GSI](https://ieeexplore.ieee.org/document/9101348).

---

## Contact

For issues or collaboration requests, please open an issue or contact the maintainer directly.

