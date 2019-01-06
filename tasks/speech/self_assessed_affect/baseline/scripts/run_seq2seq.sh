#!/bin/bash

### This Baseline is going to be a simple sequence model on the entire wavefile
base_dir=`pwd`/../
echo "Base directory is " $base_dir


## Source stuff
source ${base_dir}/path.sh
source ${base_dir}/cmd.sh

## Populate the data directory
# Filenames 
cat ${data_dir}/lab/ComParE2018_SelfAssessedAffect.tsv | grep "train" | awk '{print $1}' | cut -d '.' -f 1 > ${base_dir}/data/train/filenames 
cat ${data_dir}/lab/ComParE2018_SelfAssessedAffect.tsv | grep "devel" | awk '{print $1}' | cut -d '.' -f 1 > ${base_dir}/data/val/filenames

# Class 
cat ${data_dir}/lab/ComParE2018_SelfAssessedAffect.tsv | grep "train" | awk '{print $2}' > ${base_dir}/data/train/label
cat ${data_dir}/lab/ComParE2018_SelfAssessedAffect.tsv | grep "devel" | awk '{print $2}' > ${base_dir}/data/val/label

## Extract some form of feature representation

## Build a baseline model

