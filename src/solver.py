#!/usr/bin/env python
# encoding: utf-8

import theano
import theano.tensor as T
import numpy as np
from net import Net

# tensor
x =  T.tensor4('x')
x_mask = T.fmatrix('x_mask')
y = T.imatrix('y')
y_clip = T.ivector('y_clip')

# setting parameters
batch_size = 32
channel = 1
height = 28
n_steps = 50
lstm_hidden_units = 20
n_classes = 96
max_label_len = 20

# network structure
options = dict()
options['n_in_lstm_layer'] = height
options['n_out_lstm_layer'] = lstm_hidden_units
options['n_out_hidden_layer'] = n_classes

# build the model
net = Net(x = x, x_mask = x_mask, y = y, y_clip = y_clip, options = options)
f = theano.function([x, x_mask, y, y_clip], [net.loss, net.softmax_matrix, net.prob])

# sythenize data
x_data = np.random.rand(batch_size, channel, height, n_steps).astype('float32')
x_mask_data = np.random.randint(2, size=(batch_size, n_steps)).astype('float32')
y_data = np.random.randint(n_classes, size=(batch_size, max_label_len)).astype('int32')
y_clip_data = np.random.randint(2, max_label_len, size=(batch_size)).astype('int32')

# evaluation
loss, softmax_matrix, prob= f(x_data, x_mask_data, y_data, y_clip_data)
print("loss: ", loss)
print("softmax_matrix:", softmax_matrix)
print("prob: ", prob)
