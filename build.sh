#!/bin/sh

# Setting up variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

git submodule update --init

# echo "[*] Building VF3"
# cd $SCRIPT_DIR/benchmarks/vf3
# make

echo "[*] Building GSI"
cd $SCRIPT_DIR/benchmarks/GSI
git apply ../GSI.patch
rm objs/*.o
make

echo "[*] Building CuTS"
cd $SCRIPT_DIR/benchmarks/cuTS
bash $SCRIPT_DIR/benchmarks/cuTS/build.sh 

# echo "[*] Building MSM"
# cd $SCRIPT_DIR/benchmarks/msm
# TODO