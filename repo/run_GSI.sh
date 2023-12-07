#!/bin/bash

# Setting up variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$SCRIPT_DIR/../data/GSI
GSI_DIR=$SCRIPT_DIR/GSI
OUT_DIR=$SCRIPT_DIR/../out
EXEC_FILE=$GSI_DIR/GSI.exe

mkdir -p $OUT_DIR
mkdir -p $OUT_DIR/GSI

rm -f $OUT_DIR/GSI/GSI.out

for data in $(ls $DATA_DIR/data)
do
  for query in $(ls $DATA_DIR/query)
  do
    echo "$data - $query" >> $OUT_DIR/GSI/GSI.out
    out=$($EXEC_FILE $DATA_DIR/data/$data $DATA_DIR/query/$query)
    echo "$out" >> $OUT_DIR/GSI/GSI.out
  done
done

$SCRIPT_DIR/../scripts/parser/parseGSI.py $OUT_DIR/GSI/GSI.out $OUT_DIR/GSI/GSI.csv