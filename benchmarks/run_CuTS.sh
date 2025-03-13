# Setting up variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$SCRIPT_DIR/../data/CuTS
CUTS_DIR=$SCRIPT_DIR/cuTS
OUT_DIR=$SCRIPT_DIR/../out
EXEC_FILE=$CUTS_DIR/build/cuts

mkdir -p $OUT_DIR
mkdir -p $OUT_DIR/CuTS

rm -f $OUT_DIR/CuTS/CuTS.log

total=$(ls $DATA_DIR/query | wc -l)
i=0

for query in $(ls $DATA_DIR/query)
do
  printf "\rProgress: %d/%d" $i $total
  i=$((i+1))
  echo "$query" >> $OUT_DIR/CuTS/CuTS.log
  timeout 20 $EXEC_FILE $DATA_DIR/data.dat $DATA_DIR/query/$query >> $OUT_DIR/CuTS/CuTS.log
done

python3 $SCRIPT_DIR/../scripts/parser.py -f CuTS -o $OUT_DIR/CuTS/CuTS.csv $OUT_DIR/CuTS/CuTS.log
python3 $SCRIPT_DIR/../scripts/parser.py -f CuTS $OUT_DIR/CuTS/CuTS.log > $OUT_DIR/CuTS/CuTS.txt