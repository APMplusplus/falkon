#!/bin/bash

### This Baseline is going to be a simple raw waveform vocoder inspired by wavenet


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

# Extract some form of feature representation
if [ ! -f ${base_dir}/feats/.featextraction.done ]; then

  for d in train val
    do
      echo " Extracting features for " $d
      cat ${base_dir}/etc/filenames.$d.tdd | awk '{print $1}' > filenames.$d.tmp
  ${FALCON_DIR}/src/sigproc/do_world parallel wav2world_file ${data_dir}/wav/ filenames.$d.tmp ../feats/world_feats_20msec || exit 0
      rm -rf filenames.$d.tmp tmpdir
    done
  touch ${base_dir}/feats/.featextraction.done
fi

## Build a baseline model - Make it choose the type of model to build based on hparams file
python3.5 ${base_dir}/local/vocoding/baseline_vocoder.py

