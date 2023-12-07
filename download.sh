#!/bin/bash

git submodule update --init --recursive
cd repo
wget https://zenodo.org/records/5154114/files/CuTS-master.zip?download=1
unzip CuTS-master.zip?download=1
rm CuTS-master.zip?download=1
