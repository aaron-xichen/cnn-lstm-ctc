#!/usr/bin/env python
# encoding: utf-8
import theano.tensor as T
from lstm_layer import LSTMLayer
from common_layers import HiddenLayer
from ctc_layer import CTCLayer

class Net():
    # x is 4d tensor, (batch_size, channels, height, width)
    def __init__(self, x, x_mask, y, y_clip, options):
        self.layers = []
        self.params = dict()

        # LSTM layer
        lstm = LSTMLayer(x = x, x_mask = x_mask,
                n_features = options['n_in_lstm_layer'],
                n_hidden_units = options['n_out_lstm_layer'])
        self.layers.append(lstm)
        self.params.update(lstm.params)

        # Hidden layer with softmax activation function
        h1 = HiddenLayer(input = lstm.output,
                n_in = options['n_out_lstm_layer'],
                n_out = options['n_out_hidden_layer'],
                activation = T.nnet.softmax)
        self.layers.append(h1)
        self.params.update(h1.params)

        # CTC loss layer
        ctc = CTCLayer(x = h1.output, x_mask = x_mask, y = y, y_clip = y_clip,
                blank = -1)
        self.softmax_matrix = h1.output
        self.loss = ctc.loss
        self.debug = ctc.debug
        self.prob = ctc.prob

