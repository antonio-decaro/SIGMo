# Setting up variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$SCRIPT_DIR/../data/cuTS
CUTS_DIR=$SCRIPT_DIR/cuTS
OUT_DIR=$SCRIPT_DIR/../out
EXEC_FILE=$CUTS_DIR/build/cuts

mkdir -p $OUT_DIR
mkdir -p $OUT_DIR/cuTS

rm -f $OUT_DIR/cuTS/cuTS.log

for query in $(ls $DATA_DIR/query)
do
  echo "$query" >> $OUT_DIR/cuTS/cuTS.log
  $EXEC_FILE $DATA_DIR/data.dat $DATA_DIR/query/$query >> $OUT_DIR/cuTS/cuTS.log
done
