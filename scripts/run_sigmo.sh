#!/bin/bash
# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR=$1
shift

EXPERIMENTS="gpu-scale,sota,gpu-usage"
experiments=""
total_iterations=7

function help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -e, --experiments <exp1,exp2,...>  Comma-separated list of experiments to run (default: $EXPERIMENTS)"
  echo "  -i, --iterations <num>            Number of iterations for each experiment (default: $total_iterations)"
  echo "  -h, --help                        Display this help message"
}

# Parsing arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--experiments)
      experiments="$2"
      shift
      shift
      ;;
    -i|--iterations)
      total_iterations="$2"
      shift
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

if [[ $experiments == *"sota"* ]]; then
  echo "Running SOTA experiments..."
  OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/sota"
  mkdir -p $OUT_DIR

  for i in $(seq 0 $total_iterations); do
    printf "Iteration %i/%s\n" $i $total_iterations
    $SCRIPT_DIR/build/sigmo -i $i -Q $SCRIPT_DIR/data/SIGMO/query.dat -D $SCRIPT_DIR/data/SIGMO/data.dat -p -c query > $OUT_DIR/sigmo_${i}.log 2> $OUT_DIR/err_sigmo_${i}.log
    $SCRIPT_DIR/build/sigmo -i $i -Q $SCRIPT_DIR/data/SIGMO/query.dat -D $SCRIPT_DIR/data/SIGMO/data.dat -p -c query --find-all > $OUT_DIR/sigmo_findall_${i}.log 2> $OUT_DIR/err_sigmo_findall_${i}.log
  done
fi


if [[ $experiments == *"gpu-scale"* ]]; then
  echo "Running single GPU experiments..."
  OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/gpu_scale"
  mkdir -p $OUT_DIR
  for k in {1..25}
  do
    rm -f $OUT_DIR/logs_$k.log
    rm -f $OUT_DIR/logs_findall_$k.log
    rm -f $OUT_DIR/err_$k.log
    rm -f $OUT_DIR/err_findall_$k.log
    for i in {1..5}
    do
      $SCRIPT_DIR/build/sigmo -i 5 -c query -Q $SCRIPT_DIR/data/SIGMO/query_nowildcards.dat -D $SCRIPT_DIR/data/SIGMO/data.dat --find-all --mul-data=$k --skip-candidates-analysis >> $OUT_DIR/logs_findall_$k.log 2>> $OUT_DIR/err_findall_$k.log
      $SCRIPT_DIR/build/sigmo -i 5 -c query -Q $SCRIPT_DIR/data/SIGMO/query_nowildcards.dat -D $SCRIPT_DIR/data/SIGMO/data.dat --mul-data=$k --skip-candidates-analysis >> $OUT_DIR/logs_$k.log 2>> $OUT_DIR/err_findall_$k.log
    done
  done
fi

if [[ $experiments == *"gpu-usage"* ]]; then
  echo "Running GPU usage experiments..."
  OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/gpu_usage"
  mkdir -p $OUT_DIR

  # Start dcgmi in the background and save its PID
  dcgmi dmon -e 1003 -d 10 > $OUT_DIR/dcgmi.log 2> $OUT_DIR/err_dcgmi.log &
  DCGMI_PID=$!

  $SCRIPT_DIR/build/sigmo -i 5 -c query -Q $SCRIPT_DIR/data/SIGMO/query.dat -D $SCRIPT_DIR/data/SIGMO/data.dat --find-all --join-work-group=64 --skip-candidates-analysis > $OUT_DIR/sigmo.log 2> $OUT_DIR/err_sigmo.log
  
  echo "SIGMO PID: $!"
  kill $DCGMI_PID
  wait $DCGMI_PID 2>/dev/null

  ncu --set full -f -o $OUT_DIR/sigmo $SCRIPT_DIR/build/sigmo -i 5 -c query -Q $SCRIPT_DIR/data/SIGMO/query.dat -D $SCRIPT_DIR/data/SIGMO/data.dat --join-work-group=64 --skip-candidates-analysis
  ncu -i $OUT_DIR/sigmo.ncu-rep --print-details all --csv --print-metric-name label-name > $OUT_DIR/metrics.csv
fi