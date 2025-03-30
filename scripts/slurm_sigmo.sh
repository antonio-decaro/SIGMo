#!/bin/bash
#SBATCH -A IscrC_NETTUNE_0
#SBATCH -p boost_usr_prod
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:4
#SBATCH --job-name=sigmo_mpi
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

# Arguments
NUM_NODES=$1
NTASKS_PER_NODE=4

# Check if the number of nodes is provided
if [ -z "$NUM_NODES" ]; then
    echo "Usage: $0 <num_nodes>"
    exit 1
fi
# Check if the number of nodes is a valid integer
if ! [[ "$NUM_NODES" =~ ^[0-9]+$ ]]; then
    echo "Error: <num_nodes> must be a positive integer."
    exit 1
fi

# Reconfigure dynamic allocations
scontrol update JobId=$SLURM_JOB_ID NumNodes=$NUM_NODES

echo "Running scalability test with $NUM_NODES nodes and $NTASKS_PER_NODE tasks per node"

srun --nodes=$NUM_NODES --ntasks-per-node=$NTASKS_PER_NODE --gpus-per-task=1 --export=ALL,ONEAPI_DEVICE_SELECTOR=ext_oneapi_cuda:${SLURM_LOCALID} \
    ./build/sigmo_mpi -i 5 -Q ./data/SIGMO/query.dat -D /leonardo_scratch/large/userexternal/adecaro0/all.graph --max-data-graphs=200000 > logs/out_${N}N.log 2> logs/err_${N}N.log