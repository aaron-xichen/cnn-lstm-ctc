#!/usr/bin/env python
# encoding: utf-8

import theano.tensor as T
from common_layers import HiddenLayer
from ctc_layer import CTCLayer

class Net():
    # x is 4d tensor, (batch_size, channels, height, width)
    def __init__(self, x, x_mask, y, y_clip, options, mid_layer_type = None, forget=True):
        # self.layers = []
        self.params = dict()
        assert mid_layer_type is not None

        self.x = x
        mid = mid_layer_type(x = x, x_mask = x_mask,
                n_samples_const = options['batch_size'],
                n_features = options['n_in_lstm_layer'],
                n_hidden_units = options['n_out_lstm_layer'],
                forget = forget)

        # self.layers.append(mid)
        self.params.update(mid.params)
        self.mid_w = mid.param_w
        self.mid_b = mid.param_b
        # self.x_prime = mid.x_prime
        self.mid_output = mid.output

        # Hidden layer with softmax activation function
        h1 = HiddenLayer(input = mid.output,
                n_in = mid.n_out,
                n_out = options['n_out_hidden_layer'],
                activation = T.nnet.softmax)
        self.h_w= h1.W
        self.h_b = h1.b
        # self.layers.append(h1)
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
