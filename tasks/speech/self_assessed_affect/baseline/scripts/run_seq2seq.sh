#!/bin/bash

### This Baseline is going to be a simple sequence model on the features from entire wavefile


## Source stuff

source ../etc/path.sh
source ../etc/cmd.sh

echo "Base dir " ${base_dir}
echo "data dir " ${data_dir}

## Populate the etc directory

if [ ! -f ${base_dir}/etc/.dataprep.done ]; then

  rm -rf ${base_dir}/etc/filenames.train.tdd ${base_dir}/etc/filenames.val.tdd
  cat ${data_dir}/lab/ComParE2018_SelfAssessedAffect.tsv | while read line
    do
      fname=`echo $line | awk '{print $1}' | cut -d '.' -f 1`
      class=`echo $line | awk '{print $2}'`

      if [[ $fname == *"train"* ]]; then
         echo $fname " " ${data_dir}/wav/${fname}.wav >> ${base_dir}/etc/filenames.train.tdd
         echo $fname " " ${class} >> ${base_dir}/etc/labels.train.tdd
      elif [[ $fname == *"devel"* ]]; then
         echo $fname " " ${data_dir}/wav/${fname}.wav >> ${base_dir}/etc/filenames.val.tdd
         echo $fname " " ${class} >> ${base_dir}/etc/labels.val.tdd
      else
         echo " This is being ignored " $line
      fi
    done
  touch ${base_dir}/etc/.dataprep.done

fi


## Extract some form of feature representation

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

## Build a baseline model
python3.5 ${base_dir}/local/seqmodels/baseline.py

