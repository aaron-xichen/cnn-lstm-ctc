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
x_data_train, x_mask_data_train, y_data_train, y_clip_data_train, height, chars = prepare_training_data(
        file_path = os.path.expanduser('~/Documents/dataset/cnn-lstm-ctc/small.pkl'),
        is_shared= True,
        is_shuffle = True)
x_data_test, x_mask_data_test, y_data_test, y_clip_data_test = prepare_testing_data(is_shuffle = True)

# build tensor
print("building symbolic tensors({})".format(time.time() - begin))
x =  T.tensor4('x')
x_mask = T.matrix('x_mask')
y = T.imatrix('y')
y_clip = T.ivector('y_clip')
index = T.lscalar('index')

# setting parameters
print("setting parameters({})".format(time.time() - begin))
batch_size = 64
lstm_hidden_units = 90
n_classes = len(chars)
print("n_classes: ", n_classes)
learning_rate = 0.01
momentum = None
n_epochs = 100
resume_path = None

# compute samples num and iter
n_train_samples = len(x_data_train.get_value())
n_test_samples = len(x_data_test.get_value())
n_train_iter = n_train_samples // batch_size
n_test_iter = n_test_samples // batch_size

# network configuration
options = dict()
options['n_in_lstm_layer'] = height
options['n_out_lstm_layer'] = lstm_hidden_units
options['n_out_hidden_layer'] = n_classes + 1 # additional class blank
options['blank'] = n_classes
options['labels_len'] = y_data_train.get_value().shape[1]
options['batch_size'] = batch_size

# build the model
print("building the model({})".format(time.time() - begin))
net = Net(x = x, x_mask = x_mask, y = y, y_clip = y_clip, options = options,
        mid_layer_type = BLSTMLayer, forget=False)

# compute the grad
print("computing updates and function({})".format(time.time() - begin))
updates = []
if momentum is not None:
    assert momentum > 0 and momentum < 1
    print("using momentum:{} and learning_rate:{}".format(momentum, learning_rate))
    for name, param in net.params.items():
        m = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        v = momentum * m - learning_rate * T.grad(net.loss, param)
        updates.append((m, v))
        updates.append((param, param + v))
else:
    print("using normal sgd and learning_rate:{}".format(learning_rate))
    for name, param in net.params.items():
        print(name, type(param))
        grad = T.grad(net.loss, param)
        updates.append((param, param - learning_rate * grad))

# resume model
if resume_path is not None:
    resume_model(resume_path, net)

# build train function
print("building training function({})".format(time.time() - begin))
train  = theano.function(
        inputs = [index],
        outputs = net.loss,
        updates = updates,
        givens = {
            x : x_data_train[index * batch_size : (index + 1) * batch_size],
            x_mask : x_mask_data_train[index * batch_size : (index + 1) * batch_size],
            y : y_data_train[index * batch_size : (index + 1) * batch_size],
            y_clip : y_clip_data_train[index * batch_size : (index + 1) * batch_size]
            }
        )

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

# turn on
print("begin to train({})".format(time.time() - begin))
for epoch in range(n_epochs):
    print(".epoch {}/{} begin({:0.3f})".format(epoch+1, n_epochs, time.time() - begin))
    train_begin = time.time()
    for i in range(n_train_iter):
        loss = train(i)
        print("..loss: {}, iter:{}/{}({:0.3f})".format(loss, i+1, n_train_iter, time.time() - train_begin))
        if np.isnan(loss) or np.isinf(loss):
            print("..detect nan")
            print("..loss: {}, iter:{}/{}({:0.3f})".format(loss, i+1, n_train_iter, time.time() - train_begin))
            sys.exit()

    snapshot_path = "../snapshot/{}.pkl".format(epoch)
    snapshot(snapshot_path, net)
    test_begin = time.time()
    print(".epoch done, testing({:0.3f})".format(test_begin - train_begin))
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
    print(".testing done({:0.3f})".format(time.time()- test_begin))
    for i in range(min(10, batch_size)):
        print(".seen: {}, predict: {}".format(seqs_gt[i], seqs_pred[i]))
    accuracy = np.sum(accs) * 1.0 / len(accs)
    print(".average accuracy:{}".format(accuracy))
print("all done({:0.3f})".format(time.time() - begin))
