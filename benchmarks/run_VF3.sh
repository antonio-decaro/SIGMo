#!/bin/bash

# Setting up variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$SCRIPT_DIR/../data/VF3
VF3_DIR=$SCRIPT_DIR/vf3
OUT_DIR=$SCRIPT_DIR/../out
EXEC_FILE=$VF3_DIR/bin/vf3l

mkdir -p $OUT_DIR
mkdir -p $OUT_DIR/VF3

rm -f $OUT_DIR/VF3/VF3.log

# Count the number of query files
num_files=$(ls -1 $DATA_DIR/query/ | wc -l)
current_file=0

for query in $(ls $DATA_DIR/query)
do
  echo "$query" >> $OUT_DIR/VF3/VF3.log
  $EXEC_FILE $DATA_DIR/query/$query $DATA_DIR/data.dat -v >> $OUT_DIR/VF3/VF3.log
  printf "\rProgress: %d / %d" $((++current_file)) $num_files
done

python3 $SCRIPT_DIR/../scripts/parser.py -f VF3 -o $OUT_DIR/VF3/VF3.csv $OUT_DIR/VF3/VF3.out
python3 $SCRIPT_DIR/../scripts/parser.py -f VF3 $OUT_DIR/VF3/VF3.out > $OUT_DIR/VF3/VF3.txt