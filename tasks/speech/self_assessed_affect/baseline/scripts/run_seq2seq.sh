#!/bin/bash

### This Baseline is going to be a simple sequence model on the entire wavefile
base_dir=`pwd`/../
echo "Base directory is " $base_dir


## Source stuff

source ${base_dir}/etc/path.sh
source ${base_dir}/etc/cmd.sh

## Populate the data directory

# Filenames 
cat ${data_dir}/lab/ComParE2018_SelfAssessedAffect.tsv | grep "train" | awk '{print $1}' | cut -d '.' -f 1 > ${base_dir}/etc/filenames.train.tdd
cat ${data_dir}/lab/ComParE2018_SelfAssessedAffect.tsv | grep "devel" | awk '{print $1}' | cut -d '.' -f 1 > ${base_dir}/etc/filenames.val.tdd

# Class 
cat ${data_dir}/lab/ComParE2018_SelfAssessedAffect.tsv | grep "train" | awk '{print $2}' > ${base_dir}/etc/classids.train.tdd
cat ${data_dir}/lab/ComParE2018_SelfAssessedAffect.tsv | grep "devel" | awk '{print $2}' > ${base_dir}/data/val/classids.val.tdd

## Extract some form of feature representation

## Build a baseline model

