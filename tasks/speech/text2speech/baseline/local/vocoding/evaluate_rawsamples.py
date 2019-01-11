import numpy as np
from utils import *
from scipy.io import wavfile

## Locations
FALCON_DIR = os.environ.get('FALCON_DIR')
BASE_DIR = os.environ.get('base_dir')
DATA_DIR = os.environ.get('data_dir')
EXP_DIR = os.environ.get('exp_dir')
assert ( all (['FALCON_DIR', 'BASE_DIR', 'DATA_DIR', 'EXP_DIR']) is not None)

SAMPLES_DIR = EXP_DIR + '/samples'

'''
Original wav files will be in DATA_DIR/wav
Generated wav files will be in SAMPLES_DIR
By default lets pick a random file and print its score

original_files = sorted(os.listdir(DATA_DIR + '/wav'))
idx = np.random.choice(len(original_files))
selected_file = original_files[idx]
original_wavfile = DATA_DIR + '/wav/' + selected_file
synthesized_wavfile = SAMPLES_DIR + '/' + selected_file

print(original_wavfile, synthesized_wavfile)

'''


original_npyfile = '/home/srallaba/projects/tts/repos/wavenet_vocoder/data/cmu_arctic-audio-00001.npy'
synthesized_wavfile = '/home/srallaba/development/repos/wavenet_vocoder/test_synth/cmu_arctic-audio-00009.wav'

original = np.load(original_npyfile)
fs, synthesized = wavfile.read(synthesized_wavfile)
synthesized = mulaw_quantize(synthesized)
print(len(original), len(synthesized))

utterance_bleu_unigram = return_utterance_bleu(original, synthesized, weights=(1,0,0,0))
print("Unigram match: ", float(utterance_bleu_unigram) * 100)

utterance_bleu_bigram = return_utterance_bleu(original, synthesized, weights=(0,1,0,0))
print("Bigram match: ", float(utterance_bleu_bigram) * 100)

utterance_bleu_trigram = return_utterance_bleu(original, synthesized, weights=(0,0,1,0))
print("Trigram match: ", float(utterance_bleu_trigram) * 100)

utterance_bleu_4gram = return_utterance_bleu(original, synthesized, weights=(0,0,0,1))
print("4gram match: ", float(utterance_bleu_4gram) * 100)

