# falkon
Towards an ecosystem of tasks related to Language Technologies. Inspired by [Google Research](https://arxiv.org/ftp/arxiv/papers/1702/1702.01715.pdf). 
This repo combines design principles from Kaldi(https://github.com/kaldi-asr/kaldi) and festvox(https://github.com/festvox/festvox) but has quirks of its own.

This repo is mostly for my (and peer group's) learning. If you want SoTAs in NLP, check out [NLP Progress](https://github.com/sebastianruder/NLP-progress)

- Each page has a reading ( + implementation) list of its own. For example, list [for Emotion Recognition from Speech](https://github.com/APMplusplus/falkon/blob/master/tasks/speech/self_assessed_affect/baseline/README.md)

The goal is to make it easier to build and compare against baselines across tasks.
Since there are many tasks, it might not be feasible to put all dependencies. Two alternatives: (1) Use a virtual env for each task like AWS (2) Put a docker image

### Feel free to contribute. Easiest ways to get started are:
(1) Picking up a task (from below) or one of your choice and adding it. 
(2) Picking up an issue and working on it.

Tasks:
- [X] Self Assessed Affect Detection from Speech [Week 01]
- [X] Image Captioning [Week 01]
- [ ] Automatic Speech Recognition [Week 04]
- [ ] Atypical Emotion Recognition from Speech
- [ ] Cry Classification
- [X] Speech Synthesis [Week 01]
- [ ] Speaker Diarization
- [ ] Machine Translation
- [X] Visual Question Answering [Week 05]
- [ ] Voice Conversion
- [X] Spoofing Detection from Speech [Week 02]
- [ ] Sentiment Analysis
- [ ] Named Entity Recognition
- [ ] Toxicity Detection from Text
- [ ] Multi Target Speaker Detection and Identification
- [ ] Morphological Inflection 
- [ ] Speech Feature Extraction
- [X] Source Separation [Week 04]
- [X] Speech Enhancement [Week 04]

Concepts:
- [X] Disentanglement[Week 07]

Paradigms:
- [X] Variational Inference[Week 16]

Layers -> Modules -> Models

For example,

Conv1d++ class is a layer that enables temporal convolutions during eval.<br>
ResidualDilatedCausalConv1d is a module built on top of Conv1d++ <br>
Wavenet is a model built on top of ResidualDilatedCausalConv1d

LSTM++ class is a layer that enables learning initial hidden states based on condition. <br>
VariationalEncoderDecoder is a module built on top of LSTM++ <br>
ImageCaptioning is a model built on top of VariationalEncoderDecoder


src.nn hosts all of these. 

The directoy 'tasks' contains the individual tasks. Updated a sample speech task. I have other pressing things and so the timeline on this repo looks like end of Summer 2020. 
