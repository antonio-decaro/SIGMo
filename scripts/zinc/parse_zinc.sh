#! /bin/bash
#SBATCH --account=
#SBATCH --partition=
#SBATCH --time=04:00:00
#SBATCH --array=0-287
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --output=logs/parse_%A_%a.out
#SBATCH --error=logs/parse_%A_%a.err
#SBATCH --job-name=parse_zinc

$SCRIPT_DIR=$(dirname "$(realpath "$0")")
echo "Running parse_zinc.sh in directory: $SCRIPT_DIR"

# srun $SCRIPT_DIR/parse_zinc.py ${SLURM_ARRAY_TASK_ID}