#!/usr/bin/env python
# encoding: utf-8

import theano
import theano.tensor as T
import numpy as np
from ctc_layer import CTCLayer

x =  T.tensor3('x')
x_mask = T.fmatrix('x_mask')
y = T.imatrix('y')
y_clip = T.ivector('y_clip')

n_steps = 50
n_softmax = 128
n_samples = 30
labels_len = 20

ctc_layer = CTCLayer(x= x, x_mask = x_mask, y = y, y_clip = y_clip, blank = -1)
f = theano.function([x, x_mask, y, y_clip], [ctc_layer.cost, ctc_layer.prob, ctc_layer.debug])

x_data = np.random.rand(n_steps, n_samples, n_softmax).astype('float32')
x_mask_data = np.random.randint(2, size=(n_samples, n_steps)).astype('float32')
y_data = np.random.randint(10, n_softmax, size=(n_samples, labels_len)).astype('int32')
y_clip_data = np.random.randint(1, labels_len, size=(n_samples)).astype('int32')

cost, prob, debug = f(x_data, x_mask_data, y_data, y_clip_data)
print(cost)
print(debug[-1])
