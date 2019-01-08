#!/bin/bash

### This Baseline is going to be a simple sequence model on the features from entire wavefile


## Source stuff

. ../etc/path.sh || exit 0
. ../etc/cmd.sh || exit 0

echo "Base dir " ${base_dir}
echo "data dir " ${data_dir}

# check if the data directory eists
if [ ! -d "${data_dir}" ]; then
   echo "Seems Data not found"
   exit 0
fi

python3.5 ${base_dir}/local/baseline_lstmpacked.py
