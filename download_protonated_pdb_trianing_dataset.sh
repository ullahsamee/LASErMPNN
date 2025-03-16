#!/usr/bin/env bash

cd ./databases/

# TODO: upload laser dataset to zenodo.
wget https://zenodo.org/records/15035128/files/all_data.zip

unzip all_data.zip

cd ..
