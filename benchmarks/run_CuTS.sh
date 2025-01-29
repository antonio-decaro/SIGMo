# Setting up variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$SCRIPT_DIR/../data/CuTS
CUTS_DIR=$SCRIPT_DIR/CuTS
OUT_DIR=$SCRIPT_DIR/../out
EXEC_FILE=$CUTS_DIR/build/cuts

mkdir -p $OUT_DIR
mkdir -p $OUT_DIR/CuTS

rm -f $OUT_DIR/CuTS/CuTS.out

for data in $(ls $DATA_DIR/data)
do
  for query in $(ls $DATA_DIR/query)
  do
    echo "Running CuTS on $data - $query"
    out=$($EXEC_FILE $DATA_DIR/data/$data $DATA_DIR/query/$query)
    echo "$out" >> $OUT_DIR/CuTS/CuTS.out
    echo "" >> $OUT_DIR/CuTS/CuTS.out
  done
done