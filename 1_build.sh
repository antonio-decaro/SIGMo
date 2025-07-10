#!/bin/bash
# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
AVAILABLE_BENCHMARKS="VF3,CuTS,GSI,SIGMO"

benchmarks=$AVAILABLE_BENCHMARKS
sigmo_arch="nvidia_gpu_sm_70"
sigmo_compiler="icpx"

help()
{
    echo "Usage: ./init_msm.sh 
      [ -b=bench1,bench2,bench3] The set of benchmark files to be generated;
      [ --sigmo-arch= ] The target architecture for SIGMO;
      [ --sigmo-compiler= ] The compiler for SIGMO;
      [ -h | --help ] Print this help message and exit.
      The available benchmarks are: " $AVAILABLE_BENCHMARKS
}

# Parsing arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b=*)
      benchmarks="${1#*=}"
      shift
      ;;
    --sigmo-arch=*)
      sigmo_arch="${1#*=}"
      shift
      ;;
    --sigmo-compiler=*)
      sigmo_compiler="${1#*=}"
      shift
      ;;
    -h | --help)
      help
      exit 0
      ;;
    *)
      echo "Invalid argument: $1"
      help
      return 1 2>/dev/null
      exit 1
      ;;
  esac
done

# check if benchmarks is empty and valid
for bench in $(echo $benchmarks | sed "s/,/ /g")
do
  if [[ $AVAILABLE_BENCHMARKS != *"$bench"* ]]
  then
    echo "[!] Invalid benchmark: $bench"
    return 1 2>/dev/null
    exit 1
  fi
done
if [ -z "$benchmarks" ]
then
  benchmarks=$AVAILABLE_BENCHMARKS
fi

# if benchmark is cuTS GSI or VF3
if [[ $benchmarks == *"CuTS"* ]] || [[ $benchmarks == *"GSI"* ]] || [[ $benchmarks == *"VF3"* ]]
then
  echo "[*] Cloning submodules"
  if ! git submodule update --init; then
    echo "[!] git submodule failed, cloning manually"
    git clone https://github.com/appl-lab/CuTS.git ./cuTS
    git clone https://github.com/MiviaLab/vf3lib.git ./vf3
  fi
  cd $SCRIPT_DIR/benchmarks
  git clone https://github.com/pkumod/GSI ./GSI
fi

if [[ $benchmarks == *"CuTS"* ]]
then
  echo "[*] Building CuTS"
  cd $SCRIPT_DIR/benchmarks/cuTS
  bash $SCRIPT_DIR/benchmarks/cuTS/build.sh 
fi

if [[ $benchmarks == *"VF3"* ]]
then
  echo "[*] Building VF3"
  cd $SCRIPT_DIR/benchmarks/vf3
  make
  make vf3p
fi

if [[ $benchmarks == *"GSI"* ]]
then
  if ! gcc $SCRIPT_DIR/benchmarks/test/test_readline.c -o $SCRIPT_DIR/benchmarks/test/test_readline > /dev/null 2>&1; then
    echo "[!] readline library not found, building locally..."
    mkdir -p $SCRIPT_DIR/benchmarks/libs
    cd $SCRIPT_DIR/benchmarks/libs
    if [ ! -f "readline-8.2.tar.gz" ]; then
      wget https://ftp.gnu.org/gnu/readline/readline-8.2.tar.gz
    fi
    tar -xf readline-8.2.tar.gz
    cd readline-8.2
    ./configure --prefix=$SCRIPT_DIR/benchmarks/libs/readline-install
    make -j$(nproc)
    make install
    export CPATH=$SCRIPT_DIR/benchmarks/libs/readline-install/include:$CPATH
    export LD_LIBRARY_PATH=$SCRIPT_DIR/benchmarks/libs/readline-install/lib:$LD_LIBRARY_PATH
    export PKG_CONFIG_PATH=$SCRIPT_DIR/benchmarks/libs/readline-install/lib/pkgconfig:$PKG_CONFIG_PATH
    rm $SCRIPT_DIR/benchmarks/libs/readline-8.2.tar.gz
    rm -rf $SCRIPT_DIR/benchmarks/libs/readline-8.2
    cd $SCRIPT_DIR
  else 
    echo "[*] readline library found"
    rm $SCRIPT_DIR/benchmarks/test/test_readline
  fi

  echo "[*] Building GSI"
  cd $SCRIPT_DIR/benchmarks/GSI
  git apply ../GSI.patch
  rm objs/*.o
  make 
fi

if [[ $benchmarks == *"SIGMO"* ]]
then
  echo "[*] Building SIGMO"
  mkdir build
  cd build
  cmake ../library -DCMAKE_CXX_COMPILER=$sigmo_compiler -DSIGMO_TARGET_ARCHITECTURE=$sigmo_arch -DSIGMO_ENABLE_TEST=OFF -DCMAKE_BUILD_TYPE=Release -DCXXOPTS_BUILD_EXAMPLES=OFF -DCXXOPTS_BUILD_TESTS=OFF -DCXXOPTS_ENABLE_INSTALL=OFF
  cmake --build . -j
fi
