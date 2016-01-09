#!/usr/bin/env python
# encoding: utf-8

import theano
import theano.tensor as T
from net import Net
import time
from utee import Prefetcher, compute_acc, snapshot, resume_model
from lstm_layer import BLSTMLayer
import sys
import numpy as np

# begin to timming
begin = time.time()

# stride and patch_width
stride = 1
patch_width = [1]
height = 28 * np.sum(patch_width)
batch_size = 64

# loading data
imgs_dir = '/share/data_for_BLSTM_CTC/Samples_for_English/20151205/imgs/'
train_img_list = '/share/data_for_BLSTM_CTC/Samples_for_English/20151205/train_img_list.txt'
test_img_list = '/share/data_for_BLSTM_CTC/Samples_for_English/20151205/test_img_list.txt'
training_data_prefetcher = Prefetcher(train_img_list, imgs_dir, batch_size, stride, patch_width)
testing_data_prefetcher = Prefetcher(test_img_list, imgs_dir, batch_size, stride, patch_width)

# build tensor
print("building symbolic tensors({})".format(time.time() - begin))
x =  T.tensor4('x')
x_mask = T.matrix('x_mask')
y = T.imatrix('y')
y_clip = T.ivector('y_clip')

# shared cellar
x_shared = theano.shared(np.zeros((batch_size, 1, 10, 10)).astype(theano.config.floatX))
x_mask_shared = theano.shared(np.zeros((10, 10)).astype(theano.config.floatX))
y_shared = theano.shared(np.zeros((10, 50)).astype('int32'))
y_clip_shared = theano.shared(np.zeros(50).astype('int32'))


# setting parameters
print("setting parameters({})".format(time.time() - begin))
chars = training_data_prefetcher.chars
lstm_hidden_units = 90
n_classes = len(chars)
print("n_classes: ", n_classes)
learning_rate = theano.shared(np.float32(0.1))
momentum = None
n_epochs = 100
start_epoch = 0 # for snapshot
start_iters = 0
multisteps = set()
alpha = 0.1

# compute samples num and iter
n_train_samples = training_data_prefetcher.n_samples
n_test_samples = testing_data_prefetcher.n_samples
n_train_iter = n_train_samples // batch_size
n_test_iter = n_test_samples // batch_size

# network configuration
options = dict()
options['n_in_lstm_layer'] = height
options['n_out_lstm_layer'] = lstm_hidden_units
options['n_out_hidden_layer'] = n_classes + 1 # additional class blank
options['blank'] = n_classes
options['labels_len'] = 50
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
    print("using momentum:{} and learning_rate:{}".format(momentum, learning_rate.get_value()))
    for name, param in net.params.items():
        m = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        v = momentum * m - learning_rate * T.grad(net.loss, param)
        updates.append((m, v))
        updates.append((param, param + v))
else:
    print("using normal sgd and learning_rate:{}".format(learning_rate.get_value()))
    for name, param in net.params.items():
        print(name, type(param))
        grad = T.grad(net.loss, param)
        updates.append((param, param - learning_rate * grad))

# resume model
if start_epoch > 0:
    resume_path = "../snapshot/{}.pkl".format(start_epoch)
    resume_model(resume_path, net)


# build train function
print("building training function({})".format(time.time() - begin))
train  = theano.function(
        inputs = [],
        outputs = net.loss,
        updates = updates,
        givens = {
            x : x_shared,
            x_mask : x_mask_shared,
            y : y_shared,
            y_clip : y_clip_shared
            }
        )

# build test function
print("building testing function({})".format(time.time() - begin))
test = theano.function(
        inputs = [x, x_mask],
        outputs = net.pred,
        )

# turn on
print("begin to train({})".format(time.time() - begin))
for epoch in range(start_epoch + 1, n_epochs):
    print(".epoch {}/{} begin({:0.3f})".format(epoch, n_epochs, time.time() - begin))
    train_begin = time.time()
    for i in range(n_train_iter):
        start_iters = start_iters + 1
        # change learning rate
        if start_iters in multisteps:
            old_lr = learning_rate.get_value()
            learning_rate.set_value(np.float32(old_lr * alpha))
            print(".change learning rate from {} to {}".format(old_lr, learning_rate.get_value()))
        x_slice, x_mask_slice, y_slice, y_clip_slice = training_data_prefetcher.fetch_next(True)
        x_shared.set_value(x_slice)
        x_mask_shared.set_value(x_mask_slice)
        y_shared.set_value(y_slice)
        y_clip_shared.set_value(y_clip_slice)
        loss = train()
        print("..loss: {}, iter:{}/{}({}, {:0.3f}s)".format(loss, i+1, n_train_iter, start_iters, time.time() - train_begin))
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
        x_slice, x_mask_slice, y_slice, y_clip_slice = testing_data_prefetcher.fetch_next(False)
        y_pred = test(x_slice, x_mask_slice)
        print("..processed {}/{}({:0.3f})".format(i+1, n_test_iter, time.time() - test_begin))
        assert len(y_pred) == len(y_slice)
        seqs_pred_new, seqs_gt_new, accs_new, values_new = compute_acc(y_pred, y_slice, y_clip_slice, chars)
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
