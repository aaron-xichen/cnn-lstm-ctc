#!/usr/bin/env python
# encoding: utf-8

import theano
import theano.tensor as T
from net import Net
import time
from utee import prepare_training_data, prepare_testing_data, compute_acc, snapshot, resume_model
from lstm_layer import BLSTMLayer
import os
import sys
import numpy as np

# begin to timming
begin = time.time()

# loading data
print("loading data({})".format(time.time() - begin))
x_data_test, x_mask_data_test, y_data_test, y_clip_data_test, chars, height = prepare_testing_data()

# build tensor
print("building symbolic tensors({})".format(time.time() - begin))
x =  T.tensor4('x')
x_mask = T.matrix('x_mask')
y = T.imatrix('y')
y_clip = T.ivector('y_clip')
index = T.lscalar('index')

# setting parameters
print("setting parameters({})".format(time.time() - begin))
batch_size = 256
lstm_hidden_units = 90
n_classes = len(chars)
print("n_classes: ", n_classes)
start_epoch = 99

# compute samples num and iter
n_test_samples = len(x_data_test.get_value())
n_test_iter = n_test_samples // batch_size

# network configuration
options = dict()
options['n_in_lstm_layer'] = height
options['n_out_lstm_layer'] = lstm_hidden_units
options['n_out_hidden_layer'] = n_classes + 1 # additional class blank
options['blank'] = n_classes
options['batch_size'] = batch_size
options['labels_len'] = y_data_test.shape[1]

# build the model
print("building the model({})".format(time.time() - begin))
net = Net(x = x, x_mask = x_mask, y = y, y_clip = y_clip, options = options,
        mid_layer_type = BLSTMLayer, forget = False)

# resume model
if start_epoch > 0:
    resume_path = "../snapshot/{}.pkl".format(start_epoch)
    resume_model(resume_path, net)

# build test function
print("building testing function({})".format(time.time() - begin))
test = theano.function(
        inputs = [index],
        outputs = net.pred,
        givens = {
            x : x_data_test[index * batch_size : (index + 1) * batch_size],
            x_mask : x_mask_data_test[index * batch_size : (index + 1) * batch_size],
            }
        )

test_begin = time.time()
print("testing")
seqs_pred = []
seqs_gt = []
accs = []
values = []
for i in range(n_test_iter):
    y_pred = test(i)
    print("..processed {}/{}({:0.3f})".format(i+1, n_test_iter, time.time() - test_begin))
    y_gt = y_data_test[i * batch_size : (i+1) * batch_size]
    y_clip_gt = y_clip_data_test[i * batch_size : (i+1) * batch_size]
    assert len(y_pred) == len(y_gt)
    seqs_pred_new, seqs_gt_new, accs_new, values_new = compute_acc(y_pred, y_gt, y_clip_gt, chars)
    seqs_pred.extend(seqs_pred_new) # seqs_pred_new is a list
    seqs_gt.extend(seqs_gt_new) # seqs_gt_new is a list
    accs.extend(accs_new) # accs_new is a list
    values.extend(values_new) # vlues_new is a list
for i in range(min(10, batch_size)):
    print(".seen: {}, predict: {}".format(seqs_gt[i], seqs_pred[i]))
accuracy = np.sum(accs) * 1.0 / len(accs)
cost_time = time.time() - test_begin
total_samples = n_test_iter * batch_size
speed = cost_time * 1.0 /  total_samples
print(".average accuracy:{:0.4f}({:0.3f} / {} = {:0.4f} s/img)".format(accuracy, cost_time, total_samples, speed))
