#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

git submodule update --init

echo "[*] Building VF3"
cd $SCRIPT_DIR/benchmarks/vf3
make
make vf3p

echo "[*] Building GSI"
cd $SCRIPT_DIR/benchmarks/GSI
git apply ../GSI.patch
rm objs/*.o
make

# echo "[*] Building EGSM"
# cd $SCRIPT_DIR/benchmarks/EGSM
# mkdir build
# cd build
# cmake ..
# make

echo "[*] Building CuTS"
cd $SCRIPT_DIR/benchmarks/cuTS
bash $SCRIPT_DIR/benchmarks/cuTS/build.sh 
git apply ../cuTS.patch
