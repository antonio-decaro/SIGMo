#!/bin/bash
#SBATCH -A IscrC_NETTUNE_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=sigmo_mpi
#SBATCH --nodes=64                # Maximum nodes allocated for the entire job
#SBATCH --ntasks-per-node=4       # Default allocation (256 tasks total)
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=antdecaro@unisa.it

# Define the number of tasks per node you intend to use in each test
tasks_per_node=4

for N in 4 8 16 32 64; do
  # Calculate the total number of tasks for the current test
  total_tasks=$(( N * tasks_per_node ))

  srun -N $N -n $total_tasks --export=ALL,ONEAPI_DEVICE_SELECTOR=ext_oneapi_cuda:${SLURM_LOCALID} \
      ./build/sigmo_mpi -i 5 -Q ./data/SIGMO/query.dat -D /leonardo_scratch/large/userexternal/adecaro0/all.graph --max-data-graphs=200000 > logs/out_${N}N.log 2> logs/err_${N}N.log

  srun -N $N -n $total_tasks --export=ALL,ONEAPI_DEVICE_SELECTOR=ext_oneapi_cuda:${SLURM_LOCALID} \
      ./build/sigmo_mpi -i 5 -Q ./data/SIGMO/query.dat -D /leonardo_scratch/large/userexternal/adecaro0/all.graph --max-data-graphs=200000 --find-all > logs/out_${N}N_findall.log 2> logs/err_${N}N_findall.log
done