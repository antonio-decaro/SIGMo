#!/bin/sh

# Setting up variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "[*] Building VF3"
cd $SCRIPT_DIR/repo/vf3
make

echo "[*] Building GSI"
cd $SCRIPT_DIR/repo/GSI
make

echo "[*] Building CuTS"
cd $SCRIPT_DIR/repo/CuTS
source ./build.sh 