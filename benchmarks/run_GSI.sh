#!/bin/bash
# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

# Setting up variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$SCRIPT_DIR/../data/GSI
GSI_DIR=$SCRIPT_DIR/GSI
OUT_DIR=$SCRIPT_DIR/../out
EXEC_FILE=$GSI_DIR/GSI.exe

mkdir -p $OUT_DIR
mkdir -p $OUT_DIR/GSI

rm -f $OUT_DIR/GSI/GSI.log

for query in $(ls $DATA_DIR/query)
do
  echo "$query" >> $OUT_DIR/GSI/GSI.log
  $EXEC_FILE $DATA_DIR/data.dat $DATA_DIR/query/$query >> $OUT_DIR/GSI/GSI.log
done

python3 $SCRIPT_DIR/../scripts/parser.py -f GSI -o $OUT_DIR/GSI/GSI.csv $OUT_DIR/GSI/GSI.log
python3 $SCRIPT_DIR/../scripts/parser.py -f GSI $OUT_DIR/GSI/GSI.log > $OUT_DIR/GSI/GSI.txt