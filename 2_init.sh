#!/bin/bash
# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

# Setting up variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$SCRIPT_DIR/data
AVAILABLE_BENCHMARKS="VF3,CuTS,GSI,SIGMO"

data_limit=-1
query_limit=-1
benchmarks=""
no_wildcards=""
zinc=""
skip_parsing=False

help()
{
    echo "Usage: ./init_msm.sh 
      [ -b=bench1,bench2,bench3] The set of benchmark files to be generated;
      [ --data-limit= ] Limit the number of data files to be generated;
      [ --query-limit= ] Limit the number of query files to be generated;
      [ --skip-parsing=<True|False> ] Skip parsing of the single-node files (default: False);
      [ --zinc= ] Download the ZINC dataset in the specified path (careful, it is large);
      [ --no-wildcards ] Do not use wildcards in the query files;
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
    --no-wildcards)
      no_wildcards="--no-wildcards"
      shift
      ;;
    --zinc=*)
      zinc="${1#*=}"
      zinc=$(realpath "$zinc")
      shift
      ;;
    --skip-parsing)
      skip_parsing=True
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
    mkdir -p $OUT_DIR/query
    if [ "$bench" == "SIGMO" ]; then
      lines=$(head -n $limit $DATA_DIR/${type}.smarts)
      if [ "$type" == "query" ]; then
        python3 $SCRIPT_DIR/scripts/smile2graph.py -f $bench -o $OUT_DIR/query.dat <<< "$lines"
        python3 $SCRIPT_DIR/scripts/smile2graph.py -f $bench -o $OUT_DIR/query_nowildcards.dat --no-wildcards <<< "$lines"
        python3 $SCRIPT_DIR/scripts/smile2graph.py -f $bench -o $OUT_DIR/query/ --group-diameter <<< "$lines"
      else
        python3 $SCRIPT_DIR/scripts/smile2graph.py -f $bench -o $OUT_DIR/data.dat <<< "$lines"
      fi
    else
      while read -r line && [ $i -lt $limit ];
      do
        python3 $SCRIPT_DIR/scripts/smile2graph.py -f $bench -o $OUT_DIR/${type}/${type}_${i}.dat $no_wildcards <<< "$line" 2> /dev/null
        i=$((i+1))
        printf "\rProgress ($type): %d\%d" $i $total
      done < $DATA_DIR/query.smarts
    fi
  fi
}

if [ "$skip_parsing" == "False" ]; then
  echo "[*] Skipping parsing of single-node files."
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
fi

# if zinc is set, download the ZINC dataset
if [ "$zinc" != "" ]; then
  # if it's a direcotry add the trailing slash and the file name
  if [ -d "$zinc" ]; then
    zinc="$zinc/zinc.zst"
  fi
  echo "[*] Downloading ZINC dataset to $zinc ..."
  python $SCRIPT_DIR/scripts/zinc/download_zinc.py -o $zinc

  zinc_dir=$(dirname "$zinc")
  echo "[*] Decompressing ZINC dataset to $zinc_dir/zinc.dat ..."
  zstd -d $zinc -o $zinc_dir/zinc.dat
  rm -f $zinc

  check_hash=fcc2c2a95c89560721438b7f73cd0f226cd9973343b24e506c7cf6d77d9682d6
  echo "[*] Checking ZINC dataset hash..."
  if [ "$(sha256sum $zinc_dir/zinc.dat | awk '{print $1}')" != "$check_hash" ]; then
    echo "[!] ZINC dataset hash does not match!"
  fi
fi
# deactivate venv
deactivate