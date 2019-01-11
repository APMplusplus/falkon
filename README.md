# falkon
Towards an ecosystem of tasks related to Language Technologies. Inspired by [Google Research](https://arxiv.org/ftp/arxiv/papers/1702/1702.01715.pdf)

This repo combines design principles from Kaldi(https://github.com/kaldi-asr/kaldi) and festvox(https://github.com/festvox/festvox) but has quirks of its own.  

The goal is to make it easier to build and compare against baselines across tasks.
Since there are many tasks, it might not be feasible to put all dependencies. Two alternatives: (1) Use a virtual env for each task like AWS (2) Put a docker image

Tasks:
- [X] Self Assessed Affect Detection from Speech [Week 01]
- [X] Image Captioning [Week 01]
- [ ] Automatic Speech Recognition
- [ ] Atypical Emotion Recognition from Speech
- [ ] Cry Classification
- [X] Speech Synthesis [Week 01]
- [ ] Speaker Diarization
- [ ] Machine Translation
- [ ] Visual Question Answering
- [ ] Voice Conversion
- [ ] Spoofing Detection from Speech
- [ ] Sentiment Analysis
- [ ] Toxicity Detection from Text
- [ ] Multi Target Speaker Detection and Identification
- [ ] Morphological Inflection 

Layers -> Modules -> Models

For example,

Conv1d++ class is a layer that enables temporal convolutions during eval.<br>
ResidualDilatedCausalConv1d is a module built on top of Conv1d++ <br>
Wavenet is a model built on top of ResidualDilatedCausalConv1d

LSTM++ class is a layer that enables learning initial hidden states based on condition. <br>
VariationalEncoderDecoder is a module built on top of LSTM++ <br>
ImageCaptioning is a model built on top of VariationalEncoderDecoder


src.nn hosts all of these. 

The directoy 'tasks' contains the individual tasks. Updated a sample speech task. The timeline on this repo looks like end of Summer 2019 (33 weeks). 
