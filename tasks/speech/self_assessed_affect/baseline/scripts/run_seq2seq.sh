#!/bin/bash

### This Baseline is going to be a simple sequence model on the features from entire wavefile

#####################################################-*-mode:shell-script-*-
##                                                                       ##
##                     Carnegie Mellon University                        ##
##                         Copyright (c) 2005                            ##
##                        All Rights Reserved.                           ##
##                                                                       ##
##  Permission is hereby granted, free of charge, to use and distribute  ##
##  this software and its documentation without restriction, including   ##
##  without limitation the rights to use, copy, modify, merge, publish,  ##
##  distribute, sublicense, and/or sell copies of this work, and to      ##
##  permit persons to whom this work is furnished to do so, subject to   ##
##  the following conditions:                                            ##
##   1. The code must retain the above copyright notice, this list of    ##
##      conditions and the following disclaimer.                         ##
##   2. Any modifications must be clearly marked as such.                ##
##   3. Original authors' names are not deleted.                         ##
##   4. The authors' names are not used to endorse or promote products   ##
##      derived from this software without specific prior written        ##
##      permission.                                                      ##
##                                                                       ##
##  CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK         ##
##  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      ##
##  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   ##
##  SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE LIABLE      ##
##  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    ##
##  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   ##
##  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          ##
##  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       ##
##  THIS SOFTWARE.                                                       ##
##                                                                       ##
###########################################################################
##                                                                       ##
##  Setup the current directory for building a falcon baseline           ##
##                                                                       ##
###########################################################################

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

## Build a baseline model - Make it choose the type of model to build based on hparams file
python3.5 ${base_dir}/local/seqmodels/attention_MoL.py
