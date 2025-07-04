#! /bin/bash
#SBATCH --account=
#SBATCH --partition=
#SBATCH --time=04:00:00
#SBATCH --array=0-287
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --output=logs/parse_%A_%a.out
#SBATCH --error=logs/parse_%A_%a.err
#SBATCH --job-name=parse_zinc

SCRIPT_DIR=$1
DIR=$2

if [ -z "$DIR" ]; then
    echo "Usage: $0 <path_to_zinc_dataset>"
    exit 1
fi
if [ ! -d "$DIR" ]; then
    echo "Directory $DIR does not exist."
    exit 1
fi

DIR=$(realpath "$DIR")

srun $SCRIPT_DIR/scripts/zinc/parse_zinc.py ${SLURM_ARRAY_TASK_ID} "$DIR"