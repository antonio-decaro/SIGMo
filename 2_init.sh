#!/bin/bash

# Setting up variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$SCRIPT_DIR/data
AVAILABLE_BENCHMARKS="VF3,cuTS,GSI,MBSM"

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
      The available benchmarks are: " $AVAILABLE_BENCHMARKS
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
if [ ! -d "$SCRIPT_DIR/.venv" ]
then
  echo "[*] Creating virtual environment..."
  python3 -m venv $SCRIPT_DIR/.venv
  source $SCRIPT_DIR/.venv/bin/activate
  pip install -r $SCRIPT_DIR/scripts/requirements.txt
  deactivate
fi

# check if limits are set
if [ $query_limit -eq -1 ] # if query_limit is no set, generate all query files
then
  query_limit=$(wc -l < $DATA_DIR/query.smarts)
fi
if [ $data_limit -eq -1 ] # if data_limit is not set, generate all data files
then
  data_limit=$(wc -l < $DATA_DIR/data.smarts)
fi

# activate venv
source $SCRIPT_DIR/.venv/bin/activate

generate_files() {
  local bench=$1
  local type=$2
  local limit=$3
  local total=$(wc -l < $DATA_DIR/${type}.smarts)
  local i=0
  local OUT_DIR=$DATA_DIR/$bench

  if [ $limit -eq 0 ]
  then
    return
  fi
  mkdir -p $OUT_DIR

  if [ "$type" == "data" ]; then
    lines=$(head -n $limit $DATA_DIR/${type}.smarts)
    python3 $SCRIPT_DIR/scripts/smile2graph.py -f $bench -o $OUT_DIR/${type}.dat <<< "$lines"
  else
    if [ "$bench" == "MBSM" ]; then
      lines=$(head -n $limit $DATA_DIR/${type}.smarts)
      python3 $SCRIPT_DIR/scripts/smile2graph.py -f $bench -o $OUT_DIR/${type}.dat <<< "$lines"
    else
      while read -r line && [ $i -lt $limit ];
      do
        python3 $SCRIPT_DIR/scripts/smile2graph.py -f $bench -o $OUT_DIR/${type}/${type}_${i}.dat <<< "$line" 2>/dev/null
        i=$((i+1))
        printf "\rProgress ($type): %d\%d" $i $total
      done < $DATA_DIR/query.smarts
    fi
  fi
}

for bench in $(echo $benchmarks | sed "s/,/ /g")
do
  echo "[*] Generating $bench files..."
  mkdir -p $DATA_DIR/$bench

  # generate query and data files in parallel
  echo "[*] Generating query files..."
  generate_files $bench "query" $query_limit
  echo "[*] Generating data files..."
  generate_files $bench "data" $data_limit
  wait
done

# deactivate venv
deactivate