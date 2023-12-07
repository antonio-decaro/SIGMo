#!/bin/bash

git submodule update --init --recursive
cd repo
wget https://zenodo.org/records/5154114/files/CuTS-master.zip?download=1
mv CuTS-master.zip?download=1 CuTS-master.zip
unzip CuTS-master.zip
rm CuTS-master.zip
