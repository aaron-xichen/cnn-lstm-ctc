#!/usr/bin/env python
# encoding: utf-8
import theano
import theano.tensor as T
from net import Net
import time
from utee import prepare_training_data, prepare_testing_data, compute_acc
import os
import sys
# from theano.compile.nanguardmode import NanGuardMode
import numpy as np
import pickle as pkl
# from utee import _p

begin = time.time()
# loading data
print("loading data({})".format(time.time() - begin))
x_data_train, x_mask_data_train, y_data_train, y_clip_data_train, height, chars = prepare_training_data(
        file_path = os.path.expanduser('~/Documents/dataset/cnn-lstm-ctc/small.pkl'),
        is_shared= True)
x_data_test, x_mask_data_test, y_data_test, y_clip_data_test = prepare_testing_data()

# tensor
print("building symbolic tensors({})".format(time.time() - begin))
# x =  T.tensor4('x')
# x_mask = T.fmatrix('x_mask')
# y = T.imatrix('y')
# y_clip = T.ivector('y_clip')
# index = T.lscalar('index')

x =  T.tensor4('x')
x_mask = T.matrix('x_mask')
y = T.imatrix('y')
y_clip = T.ivector('y_clip')
index = T.lscalar('index')

# setting parameters
print("setting parameters({})".format(time.time() - begin))
batch_size = 32
lstm_hidden_units = 90
n_classes = len(chars)
learning_rate = 0.001
momentum = None
n_epochs = 10

# compute
n_train_samples = len(x_data_train.get_value())
n_test_samples = len(x_data_test.get_value())
n_train_iter = n_train_samples // batch_size
n_test_iter = n_test_samples // batch_size

# network structure
options = dict()
options['n_in_lstm_layer'] = height
options['n_out_lstm_layer'] = lstm_hidden_units
options['n_out_hidden_layer'] = n_classes + 1 # additional class blank
options['blank'] = n_classes
options['labels_len'] = y_data_train.get_value().shape[1]

# build the model
print("building the model({})".format(time.time() - begin))
net = Net(x = x, x_mask = x_mask, y = y, y_clip = y_clip, options = options)

# compute the grad
print("computing updates and function({})".format(time.time() - begin))
loss = net.loss
params = net.params

# weight decay, l1 norm
# loss += 5 * weight_decay * T.sum(net.params[_p('lstm', 'b')] ** 2)
# loss += weight_decay * T.sum(net.params[_p('lstm', 'W')] ** 2)
# for param in params:
    # loss += weight_decay * T.sum(param ** 2)

updates = []
grads = []
names = []
if momentum is not None:
    for name, param in params.items():
        m = theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX))
        v = momentum * m - learning_rate * T.grad(loss, param)
        updates.append((m, v))
        new_param = param + v
        updates.append((param, new_param))
else:
    for name, param in params.items():
        print(name, type(param))
        grad = T.grad(loss, param)
        grads.append(grad)
        names.append(name)
        updates.append((param, param - learning_rate * grad))

# build train function
print("building training function({})".format(time.time() - begin))
train  = theano.function(
        inputs = [index],
        outputs = [net.loss, net.debug, net.prob, net.param_w, net.param_b, net.lstm_output, net.param_hw, net.param_hb, net.pre_activation, grads[0], grads[1], grads[2], grads[3], grads[4], net.pin0, net.pin1, net.pin2],
        updates = updates,
        givens = {
            x : x_data_train[index * batch_size : (index + 1) * batch_size],
            x_mask : x_mask_data_train[index * batch_size : (index + 1) * batch_size],
            y : y_data_train[index * batch_size : (index + 1) * batch_size],
            y_clip : y_clip_data_train[index * batch_size : (index + 1) * batch_size]
            },
        # mode = NanGuardMode(nan_is_error=True, inf_is_error=True)
        )

# build test function
print("building testing function({})".format(time.time() - begin))
test = theano.function(
        inputs = [index],
        outputs = [net.param_w, net.param_b, net.x_prime, net.lstm_output, net.pre_activation, net.softmax_matrix, net.pred],
        givens = {
            x : x_data_test[index * batch_size : (index + 1) * batch_size],
            x_mask : x_mask_data_test[index * batch_size : (index + 1) * batch_size],
            }
        )

# turn on
print("begin to train, reset the timer({})".format(time.time() - begin))
begin = time.time()
all_weights = dict()
for epoch in range(n_epochs):
    print(".epoch {}/{}({:0.3f})".format(epoch+1, n_epochs, time.time() - begin))
    losses = []
    train_begin = time.time()
    for i in range(n_train_iter):
        loss, debug, prob, lstm_w, lstm_b, lstm_output, h_w, h_b, pre_activation, grad0, grad1, grad2, grad3, grad4, pin0, pin1, pin2 = train(i)
        # print("pin0:", pin0)
        # print("pin1:", pin1)
        # print("pin2:", pin2)
        print("..loss: {}, iter:{}/{}({:0.3f})".format(loss, i+1, n_train_iter, time.time() - train_begin))
        # print("hidden pre_activation: {}".format(np.mean(pre_activation)))
        # print("grad {}:{},{},{}".format(names[0], np.min(grad0), np.max(grad0), np.mean(grad0)))
        # print("grad {}:{},{},{}".format(names[1], np.min(grad1), np.max(grad1), np.mean(grad1)))
        # print("grad {}:{},{},{}".format(names[2], np.min(grad2), np.max(grad2), np.mean(grad2)))
        # print("grad {}:{},{},{}".format(names[3], np.min(grad3), np.max(grad3), np.mean(grad3)))
        # print("grad {}:{},{},{}".format(names[4], np.min(grad4), np.max(grad4), np.mean(grad4)))
        # print("lstm_w:{}, lstm_b:{}, lstm_output:{}, h_w:{}, h_b:{}".format(
            # np.sum(np.isnan(lstm_w)),
            # np.sum(np.isnan(lstm_b)),
            # np.sum(np.isnan(lstm_output)),
            # np.sum(np.isnan(h_w)),
            # np.sum(np.isnan(h_b))))
        if np.isnan(loss) or np.isinf(loss):
            print("detect nan")
            sys.exit()
        # for key, value in net.params.iteritems():
            # if not key in all_weights:
                # all_weights[key] = []
            # all_weights[key].append(np.mean(np.mean(value.get_value())))
        # losses.append(loss)
        # break
    # test_begin = time.time()
    # print("...testing({:0.3f})".format(test_begin - train_begin))
    # seqs_pred = []
    # seqs_gt = []
    # accs = []
    # values = []
    # for i in range(n_test_iter):
        # lstm_w, lstm_b, x_prime, lstm_output, pre_activation, softmax_output, y_pred = test(i)
        # # print("lstm_w: ", lstm_w)
        # # print("lstm_b: ", lstm_b)
        # # print("x_prime: ", x_prime)
        # # print("lstm_output: ", lstm_output)
        # print("y_pred: ", y_pred)
        # # print("pre_activation: ", pre_activation[0, :, :])
        # # print("softmax_output: ", softmax_output[0, :, :])
        # y_gt = y_data_test[i * batch_size : (i+1) * batch_size]
        # y_clip_gt = y_clip_data_test[i * batch_size : (i+1) * batch_size]
        # assert len(y_pred) == len(y_gt)
        # seqs_pred_new, seqs_gt_new, accs_new, values_new = compute_acc(y_pred, y_gt, y_clip_gt, chars)
        # seqs_pred.extend(seqs_pred_new) # seqs_pred_new is a list
        # seqs_gt.extend(seqs_gt_new) # seqs_gt_new is a list
        # accs.extend(accs_new) # accs_new is a list
        # values.extend(values_new) # vlues_new is a list
        # break
    # accuracy = np.sum(accs) * 1.0 / len(accs)
    # test_end = time.time()
    # for i in range(min(5, batch_size)):
        # print("seen: {}, predict: {}".format(seqs_gt[i], seqs_pred[i]))
    # print("...average accuracy:{}({:0.3f})".format(accuracy, time.time() - test_end))
print("all done({:0.3f})".format(time.time() - begin))

with open("info.pkl", 'w') as f:
    pkl.dump(all_weights, f)
