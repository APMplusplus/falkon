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
##  If the user wants to use a custom base directory, they can specify   ##
##  it with the -b tag.                                                  ##
##                                                                       ##
###########################################################################

## Source stuff

. ../etc/path.sh || exit 0
. ../etc/cmd.sh || exit 0

echo "Base dir " ${base_dir}
echo "data dir " ${data_dir}


# Check if feats directory exists
if [ ! -d "${feats_dir}" ]; then
   echo "Seems features not found"
   exit 0
fi


cat ${data_dir}/ASVspoof2019_LA_protocols/ASVspoof2019.LA.cm.train.trn.txt | awk '{print $2 " "  $5}' > ../etc/tdd.la.train || exit 0
cat ${data_dir}/ASVspoof2019_LA_protocols/ASVspoof2019.LA.cm.dev.trl.txt | awk '{print $2 " "  $5}' > ../etc/tdd.la.dev || exit 0


## Build a baseline model 
python3.5 ${base_dir}/local/super_vectors/baseline.py