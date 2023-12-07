#!/bin/bash

# Setting up variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$SCRIPT_DIR/data
AVAILABLE_BENCHMARKS="msm,vf3,GSI"

benchmarks=""

help()
{
    echo "Usage: ./init_msm.sh 
      [ -b=bench1,bench2,bench3] The set of benchmark files to be generated;
      [ --data-limit= ] Limit the number of data files to be generated;
      [ --query-limit= ] Limit the number of query files to be generated;
      [ -h | --help ] Print this help message and exit.
      The available benchmarks are: [msm, vf3, GSI]"
}

# Parsing arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b=*)
      benchmarks="${1#*=}"
      shift
      ;;
    -h | --help)
      help
      exit 0
      ;;
    *)
      echo "Invalid argument: $1"
      help
      return 1 2>/dev/null
      exit 1
      ;;
  esac
done

# check if benchmarks is empty and valid
for bench in $(echo $benchmarks | sed "s/,/ /g")
do
  if [[ $AVAILABLE_BENCHMARKS != *"$bench"* ]]
  then
    echo "[!] Invalid benchmark: $bench"
    return 1 2>/dev/null
    exit 1
  fi
done
if [ -z "$benchmarks" ]
then
  benchmarks=$AVAILABLE_BENCHMARKS
fi

for bench in $(echo $benchmarks | sed "s/,/ /g")
do
  # check if data directory exists
  if [ ! -d "$DATA_DIR/$bench" ]
  then
    echo "[!] Data directory for $bench does not exist. Please run ./init.sh first."
  else
    $SCRIPT_DIR/repo/run_$bench.sh 
  fi
done