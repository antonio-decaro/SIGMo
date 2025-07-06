#!/bin/bash
#SBATCH --account=
#SBATCH --partition=
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=sigmo_mpi
#SBATCH --output=out_%N.log
#SBATCH --error=err_%N.log
#SBATCH --time=00:10:00

SCRIPT_DIR=$1
N=$2
FIND_ALL=$3
DATA_FILE=$4
TASKS=$((N * 4))

# if findall is set to 1, use the find_all.sh script
if [ "$FIND_ALL" == "1" ]; then
  FIND_ALL="--find-all"
else
  FIND_ALL=""
fi

mpirun -n $TASKS $SCRIPT_DIR/build/sigmo_mpi -i 5 -Q $SCRIPT_DIR/data/SIGMO/query_nowildcards.dat -D $DATA_FILE --max-data-graphs=350000 $FIND_ALL