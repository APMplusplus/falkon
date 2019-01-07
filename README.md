# falkon
Towards an ecosystem of tasks related to Language Technologies

This repo combines design principles from Kaldi(https://github.com/kaldi-asr/kaldi) and festvox(https://github.com/festvox/festvox) but has quirks of its own.  

The goal is to make it easier to build and compare against baselines across tasks

Layers -> Modules -> Models

For example,

Conv1d++ class is a layer that enables temporal convolutions during eval.
ResidualDilatedCausalConv1d is a module built on top of Conv1d++
Wavenet is a model built on top of ResidualDilatedCausalConv1d

LSTM++ class is a layer that enables learning initial hidden states based on condition
VariationalEncoderDecoder is a module built on top of LSTM++
ImageCaptioning is a model built on top of VariationalEncoderDecoder


