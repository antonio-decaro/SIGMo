#!/bin/bash
# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR=$1
shift

EXPERIMENTS="core,diameter,dataset-scale,gpu-metrics"
experiments=""
total_iterations=7

function help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -e, --experiments <exp1,exp2,...>  Comma-separated list of experiments to run (default: $EXPERIMENTS)"
  echo "  -i, --iterations <num>            Number of iterations for each experiment (default: $total_iterations)"
  echo "  -H, --help                        Display this help message"
}

# Parsing arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -e=*|--experiments=*)
      experiments="${1#*=}"
      shift
      ;;
    -i=*|--iterations=*)
      total_iterations="${1#*=}"
      shift
      ;;
    -H)
      help
      exit 0
      ;;
    *)
      help
      exit 1
      ;;
  esac
done

# check if experiments is empty and valid
for exp in $(echo $experiments | sed "s/,/ /g")
do
  if [[ $EXPERIMENTS != *"$exp"* ]]
  then
    echo "[!] Invalid experiment: $exp"
    return 1 2>/dev/null
    exit 1
  fi
done
if [ -z "$experiments" ]
then
  experiments=$EXPERIMENTS
fi

start_time=$(date +%s)

if [[ $experiments == *"core"* ]]; then
  echo "Running SIGMo assessment experiments..."
  OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/core"
  mkdir -p $OUT_DIR

  for i in $(seq 0 $total_iterations); do
    printf "Iteration %i/%s\n" $i $total_iterations
    $SCRIPT_DIR/build/sigmo -i $i -Q $SCRIPT_DIR/data/SIGMO/query.dat -D $SCRIPT_DIR/data/SIGMO/data.dat -p -c query > $OUT_DIR/sigmo_${i}.log 2> $OUT_DIR/err_sigmo_${i}.log
    $SCRIPT_DIR/build/sigmo -i $i -Q $SCRIPT_DIR/data/SIGMO/query.dat -D $SCRIPT_DIR/data/SIGMO/data.dat -p -c query --find-all > $OUT_DIR/sigmo_findall_${i}.log 2> $OUT_DIR/err_sigmo_findall_${i}.log
  done

  python $SCRIPT_DIR/scripts/output_analyzer.py $OUT_DIR $SCRIPT_DIR/out/SIGMO/sigmo_results.csv
fi


if [[ $experiments == *"dataset-scale"* ]]; then
  echo "Running Dataset Scaling experiments..."
  OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/dataset_scale"
  mkdir -p $OUT_DIR
  for k in {1..25}
  do
    printf "Dataset scale %i\n" $k
    rm -f $OUT_DIR/logs_$k.log
    rm -f $OUT_DIR/logs_findall_$k.log
    rm -f $OUT_DIR/err_$k.log
    rm -f $OUT_DIR/err_findall_$k.log
    for i in {1..5}
    do
      printf "Iteration %i/%s\n" $i 5
      $SCRIPT_DIR/build/sigmo -i 5 -c query -Q $SCRIPT_DIR/data/SIGMO/query_nowildcards.dat -D $SCRIPT_DIR/data/SIGMO/data.dat --max-data-graphs 1000000000 --find-all --mul-data=$k --skip-candidates-analysis >> $OUT_DIR/logs_findall_$k.log 2>> $OUT_DIR/err_findall_$k.log
      $SCRIPT_DIR/build/sigmo -i 5 -c query -Q $SCRIPT_DIR/data/SIGMO/query_nowildcards.dat -D $SCRIPT_DIR/data/SIGMO/data.dat --max-data-graphs 1000000000 --mul-data=$k --skip-candidates-analysis >> $OUT_DIR/logs_$k.log 2>> $OUT_DIR/err_findall_$k.log
    done
  done
fi

if [[ $experiments == *"diameter"* ]]; then
  echo "Running diameter experiments..."
  OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/diameter"
  mkdir -p $OUT_DIR

  # list all the files in $SCIRPT_DIR/data/SIGMO/query/*.dat
  query_files=($SCRIPT_DIR/data/SIGMO/query/*.dat)
  

  for query_file in "${query_files[@]}"; do
    query_name=$(basename "$query_file" .dat)
    query_name=$(echo "$query_name" | sed 's/^query_//')
    for i in $(seq 0 $total_iterations); do
      printf "Query %s - Iteration %i/%s\n" "$query_name" $i $total_iterations
      $SCRIPT_DIR/build/sigmo -i $i -c query -Q "$query_file" -D $SCRIPT_DIR/data/SIGMO/data.dat --find-all --skip-candidates-analysis > $OUT_DIR/sigmo_${query_name}_${i}.log
    done
  done
fi

if [[ $experiments == *"gpu-metrics"* ]]; then
  echo "Running GPU metrics experiments..."
  OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/gpu_metrics"
  mkdir -p $OUT_DIR

  # Start dcgmi in the background and save its PID
  dcgmi dmon -e 1003 -d 10 > $OUT_DIR/dcgmi.log 2> $OUT_DIR/err_dcgmi.log &
  DCGMI_PID=$!

  $SCRIPT_DIR/build/sigmo -i 5 -c query -Q $SCRIPT_DIR/data/SIGMO/query.dat -D $SCRIPT_DIR/data/SIGMO/data.dat --find-all --join-work-group=64 --skip-candidates-analysis > $OUT_DIR/sigmo.log 2> $OUT_DIR/err_sigmo.log
  
  echo "SIGMO PID: $!"
  kill $DCGMI_PID
  wait $DCGMI_PID 2>/dev/null

  ncu --set full -f -o $OUT_DIR/sigmo $SCRIPT_DIR/build/sigmo -i 5 -c query -Q $SCRIPT_DIR/data/SIGMO/query.dat -D $SCRIPT_DIR/data/SIGMO/data.dat --join-work-group=64 --skip-candidates-analysis
  ncu -i $OUT_DIR/sigmo.ncu-rep --print-details all --csv --print-metric-name name > $OUT_DIR/metrics.csv
fi

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Experiments completed in $elapsed_time seconds."