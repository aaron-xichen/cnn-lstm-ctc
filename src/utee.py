#!/usr/bin/env python
# encoding: utf-8
import theano
from theano import config
import numpy as np
import pickle as pkl
import os
def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def shared(data, name = None):
    if name is not None:
        return theano.shared(np_floatX(data), name = name)
    else:
        return theano.shared(np_floatX(data))

def _p(pp, name):
    return '%s_%s' % (pp, name)

def prepare_data(
        file_path = os.path.expanduser('~/Documents/dataset/cnn-lstm-ctc/tiny.pkl'),
        is_shuffle = True,
        is_shared = True,
        n = None,
        channels = 1):
    with open(file_path, 'r') as f:
        data = pkl.load(f)
        chars = data['chars']
        n_classes = len(chars)
        xs = data['x']
        ys = data['y']
        assert len(xs) == len(ys)
        n_samples = len(xs)
        height = xs[0].shape[0]
        x_max_len = np.max([x.shape[1] for x in xs])
        y_max_len = np.max([2 * len(y) + 1 for y in ys])
        print("x_max_step: {}, y_max_width: {}".format(x_max_len, y_max_len))

        # x and x_mask
        x = np.zeros((n_samples, channels, height, x_max_len)). astype('float32')
        x_mask = np.zeros((n_samples, x_max_len)).astype('float32')
        for i, xx in enumerate(xs):
            shape = xx.shape
            assert height == shape[0]
            x[i, :, :, :shape[1]] = xx.astype('float32')
            x_mask[i, :shape[1]] = 1.0


        # y and y_clip
        y = np.zeros((n_samples, y_max_len)).astype('int32')
        y_clip = np.zeros((n_samples)).astype('int32')
        for i, yy in enumerate(ys):
            y_extend = np.ones(2 * len(yy) + 1, dtype='int32') * n_classes
            for j in range(len(yy)):
                y_extend[2 * j + 1] = yy[j]
            y[i, :len(y_extend)] = y_extend
            y_clip[i] = len(y_extend)

        values = [x, x_mask, y, y_clip]

        # is shuffle
        if is_shuffle:
            perms = np.random.permutation(n_samples)
            values = [value[perms] for value in values]

        # slice
        if n is not None:
            values = [value[:n] for value in values]

        # is shared
        if is_shared:
            values = [theano.shared(value) for value in values]

        values.append(height)
        values.append(chars)
        return values
        # return x.reshape(x.shape[0], 1, x.shape[1], x.shape[2]), x_mask, y, y_clip, chars
        # return theano.shared(x), theano.shared(x_mask), theano.shared(y), theano.shared(y_clip), chars
