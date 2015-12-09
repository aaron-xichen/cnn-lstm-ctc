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
        self.x = x
        lstm = LSTMLayer(x = x, x_mask = x_mask,
                n_features = options['n_in_lstm_layer'],
                n_hidden_units = options['n_out_lstm_layer'])
        self.layers.append(lstm)
        self.params.update(lstm.params)
        self.param_w = lstm.param1
        self.param_b = lstm.param2
        self.x_prime = lstm.x_prime
        self.lstm_output = lstm.output

        # Hidden layer with softmax activation function
        h1 = HiddenLayer(input = lstm.output,
                n_in = options['n_out_lstm_layer'],
                n_out = options['n_out_hidden_layer'],
                activation = T.nnet.softmax)
        self.param_hw = h1.W
        self.param_hb = h1.b
        self.layers.append(h1)
        self.params.update(h1.params)
        self.pre_activation = h1.pre_activation
        self.softmax_matrix = h1.output
        self.pred = T.argmax(self.softmax_matrix, axis = 2).T

        # CTC loss layer
        ctc = CTCLayer(x = h1.output, x_mask = x_mask, y = y, y_clip = y_clip,
                labels_len_const = options['labels_len'],
                blank = options['blank'], log_space = True)

        # return value
        self.loss = ctc.loss
        self.debug = ctc.debug
        self.prob = ctc.prob


        self.pin0 = ctc.pin0
        self.pin1 = ctc.pin1
        self.pin2 = ctc.pin2
