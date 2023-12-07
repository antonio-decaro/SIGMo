#!/bin/bash

# Setting up variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MAIN_DIR=$SCRIPT_DIR/..
AVAILABLE_BENCHMARKS="msm,vf3"

data_limit=-1
query_limit=-1
benchmarks=""

help()
{
    echo "Usage: ./init_msm.sh 
      [ -b=bench1,bench2,bench3] The set of benchmark files to be generated;
      [ --data-limit= ] Limit the number of data files to be generated;
      [ --query-limit= ] Limit the number of query files to be generated;
      [ -h | --help ] Print this help message and exit.
      The available benchmarks are: [msm, vf3]"
}

# Parsing arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-limit=*)
      data_limit="${1#*=}"
      shift
      ;;
    --query-limit=*)
      query_limit="${1#*=}"
      shift
      ;;
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

echo "Selected benchmarks: $benchmarks"

# check if venv exists
if [ ! -d "$MAIN_DIR/scripts/.venv" ]
then
  echo "[*] Creating virtual environment..."
  python3 -m venv $MAIN_DIR/scripts/.venv
  source $MAIN_DIR/scripts/.venv/bin/activate
  pip3 install -r $MAIN_DIR/scripts/requirements.txt
  deactivate
fi

# activate venv
source $MAIN_DIR/scripts/.venv/bin/activate

for bench in $(echo $benchmarks | sed "s/,/ /g")
do
  # creating output directories
  OUT_DIR=$SCRIPT_DIR/$bench
  mkdir -p $OUT_DIR
  mkdir -p $OUT_DIR/data
  mkdir -p $OUT_DIR/query

  # generate query files
  i=0
  if [ $query_limit -eq -1 ] # if query_limit is no set, generate all query files
  then
    query_limit=$(wc -l < $SCRIPT_DIR/query.smarts)
  fi
  while read -r line && [ $i -lt $query_limit ];
  do
    $MAIN_DIR/scripts/smarts2${bench}.py $line > $OUT_DIR/query/query_$i.dat
    i=$((i+1))
  done < $SCRIPT_DIR/query.smarts

  # generate data files
  i=0
  if [ $data_limit -eq -1 ] # if data_limit is not set, generate all data files
  then
    data_limit=$(wc -l < $SCRIPT_DIR/data.smarts)
  fi
  while read -r line && [ $i -lt $data_limit ];
  do
    $MAIN_DIR/scripts/smarts2${bench}.py $line > $OUT_DIR/data/data_$i.dat
    i=$((i+1))
  done < $SCRIPT_DIR/data.smarts
done

# deactivate venv
deactivate
