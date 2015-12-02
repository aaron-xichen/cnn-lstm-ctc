#!/usr/bin/env python
# encoding: utf-8
import theano
import theano.tensor as T
from net import Net
import time
from utee import prepare_data

begin = time.time()
# tensor
print("building symbolic tensors({})".format(time.time() - begin))
x =  T.tensor4('x')
x_mask = T.fmatrix('x_mask')
y = T.imatrix('y')
y_clip = T.ivector('y_clip')

# setting parameters
print("setting parameters({})".format(time.time() - begin))
batch_size =  16
channel = 1
height = 28
n_steps = 30
lstm_hidden_units = 20
n_classes = 96
max_label_len = 20
learning_rate = 0.01
n_iters = batch_size * 10

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
print("building updates and function({})".format(time.time() - begin))
loss = net.loss
params = net.params.values()
updates = []
for param in params:
    updates.append((param, param - learning_rate * T.grad(loss, param)))

f = theano.function([x, x_mask, y, y_clip],
        [net.loss, net.softmax_matrix, net.prob, net.pin],
        updates = updates)

# # sythenize data
# x_data = np.random.rand(batch_size, channel, height, n_steps).astype('float32')
# x_mask_data = np.random.randint(2, size=(batch_size, n_steps)).astype('float32')
# y_data = np.random.randint(n_classes, size=(batch_size, max_label_len)).astype('int32')
# y_clip_data = np.random.randint(2, max_label_len, size=(batch_size)).astype('int32')

# loading data
print("loading data({})".format(time.time() - begin))
x_data, x_mask_data, y_data, y_clip_data, chars = prepare_data(n = batch_size,
        is_shared= False)

# evaluation
print("begin to train({})".format(time.time() - begin))
for i in range(n_iters // batch_size):
    print("...epoch {}/{}({})".format(i+1, n_iters // batch_size, time.time() - begin))
    loss, softmax_matrix, prob, pin= f(x_data, x_mask_data, y_data, y_clip_data)
    print("loss: {}({})".format(loss, time.time() - begin))
    # print("softmax_matrix: ", softmax_matrix)
    # print("probs: ", prob)
    print("pin: ", pin)
print("all done({})".format(time.time() - begin))
# end = time.time()
# print("cost {}s with batch_size {}".format(end - begin, batch_size))
# print("loss: ", loss)
# print("softmax_matrix:", softmax_matrix)
# print("prob: ", prob)
