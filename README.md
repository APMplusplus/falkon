# falkon
Towards an ecosystem of tasks related to Language Technologies

This repo combines design principles from Kaldi(https://github.com/kaldi-asr/kaldi) and festvox(https://github.com/festvox/festvox) but has quirks of its own.  

The goal is to make it easier to build and compare against baselines across tasks.

Tasks:
- [X] Self Assessed Affect Detection from Speech
- [X] Image Captioning
- [ ] Automatic Speech Recognition
- [ ] Atypical Emotion Recognition from Speech
- [ ] Cry Classification
- [X] Speech Synthesis
- [ ] Machine Translation
- [ ] Visual Question Answering
- [ ] Voice Conversion
- [ ] Spoofing Detection from Speech
- [ ] Sentiment Analysis
- [ ] Toxicity Detection from Text
- [ ] Multi Target Speaker Detection and Identification

Layers -> Modules -> Models

For example,

Conv1d++ class is a layer that enables temporal convolutions during eval.<br>
ResidualDilatedCausalConv1d is a module built on top of Conv1d++ <br>
Wavenet is a model built on top of ResidualDilatedCausalConv1d

LSTM++ class is a layer that enables learning initial hidden states based on condition. <br>
VariationalEncoderDecoder is a module built on top of LSTM++ <br>
ImageCaptioning is a model built on top of VariationalEncoderDecoder


src.nn hosts all of these. 

The directoy 'tasks' contains the individual tasks. Updated a sample speech task. The timeline on this repo looks like end of Summer 2018. 
