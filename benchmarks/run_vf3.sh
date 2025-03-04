#!/bin/bash

# Setting up variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$SCRIPT_DIR/../data/VF3
VF3_DIR=$SCRIPT_DIR/vf3
OUT_DIR=$SCRIPT_DIR/../out
EXEC_FILE=$VF3_DIR/bin/vf3l

mkdir -p $OUT_DIR
mkdir -p $OUT_DIR/VF3

rm -f $OUT_DIR/VF3/VF3.out

for data in $(ls $DATA_DIR/data)
do
  for query in $(ls $DATA_DIR/query)
  do
    echo "$data - $query" >> $OUT_DIR/VF3/VF3.out
    out=$($EXEC_FILE $DATA_DIR/query/$query $DATA_DIR/data/$data | tail -1)
    echo "$out" >> $OUT_DIR/VF3/VF3.out
  done
done

$SCRIPT_DIR/../scripts/parser/parseVF3.py $OUT_DIR/VF3/VF3.out $OUT_DIR/VF3/VF3.csv