import os, sys
import numpy as np
import soundfile as sf
import pyworld as pw
import gzip
import pickle
import scipy.misc

file = sys.argv[1]
feats_dir = sys.argv[2]

def extract_feats(file, feats_dir):
    fname = os.path.basename(file).split('.wav')[0]
    x, fs = sf.read(file)
    f0, sp, ap = pw.wav2world(x, fs, frame_period=5) 

    np.savetxt(feats_dir + '/' + fname + '.f0_ascii', f0)
    np.savetxt(feats_dir + '/' + fname + '.sp_ascii', sp)
    np.savetxt(feats_dir + '/' + fname + '.ap_ascii', ap)
 
    #print "Saved to ", feats_dir + '/' + fname + '.f0_ascii'


extract_feats(file, feats_dir)
