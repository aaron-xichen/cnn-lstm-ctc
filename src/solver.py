#!/usr/bin/env python
# encoding: utf-8
import theano
import theano.tensor as T
from net import Net
import time
from utee import prepare_data
import os

begin = time.time()

# loading data
print("loading data({})".format(time.time() - begin))
x_data_train, x_mask_data_train, y_data_train, y_clip_data_train, height, chars = prepare_data(
        file_path = os.path.expanduser('~/Documents/dataset/cnn-lstm-ctc/small.pkl'),
        is_shared= False)

# tensor
print("building symbolic tensors({})".format(time.time() - begin))
x =  T.tensor4('x')
x_mask = T.fmatrix('x_mask')
y = T.imatrix('y')
y_clip = T.ivector('y_clip')
index = T.lscalar('index')

# setting parameters
print("setting parameters({})".format(time.time() - begin))
batch_size =  16
lstm_hidden_units = 20
n_classes = len(chars) + 1 # additional seperate char in ctc
learning_rate = 0.01
n_epochs = 1

# compute
n_train_samples = len(x_data_train.get_value())
n_train_iter = n_train_samples // batch_size

# network structure
options = dict()
options['n_in_lstm_layer'] = height
options['n_out_lstm_layer'] = lstm_hidden_units
options['n_out_hidden_layer'] = n_classes
options['blank'] = -1

# build the model
print("building the model({})".format(time.time() - begin))
net = Net(x = x, x_mask = x_mask, y = y, y_clip = y_clip, options = options)

# compute the grad
print("computing updates and function({})".format(time.time() - begin))
loss = net.loss
params = net.params.values()
updates = []
for param in params:
    updates.append((param, param - learning_rate * T.grad(loss, param)))

# build train function
print("building training function({})".format(time.time() - begin))
train  = theano.function(
        inputs = [index],
        # outputs = [net.loss, net.softmax_matrix, net.prob, net.pin],
        outputs = [net.loss],
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

# turn on
print("begin to train, reset the timer({})".format(time.time() - begin))
begin = time.time()
for epoch in range(n_epochs):
    print(".epoch {}/{}({})".format(epoch+1, n_epochs, time.time() - begin))
    losses = []
    for i in range(n_train_iter):
        loss = train(i)
        print("..loss: {}, iter:{}/{}({})".format(loss, i+1, n_train_iter, time.time() - begin))
        losses.append(loss)
print("all done({})".format(time.time() - begin))
