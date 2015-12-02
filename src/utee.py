#!/usr/bin/env python
# encoding: utf-8
import theano
from theano import config
import numpy as np
import pickle as pkl
import os
import editdistance as ed
def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def shared(data, name = None):
    if name is not None:
        return theano.shared(np_floatX(data), name = name)
    else:
        return theano.shared(np_floatX(data))

def _p(pp, name):
    return '%s_%s' % (pp, name)

def prepare_training_data(
        file_path = None,
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
        print("training, x_max_step: {}, y_max_width: {}".format(x_max_len, y_max_len))

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

def prepare_testing_data(
        file_path = os.path.expanduser('~/Documents/dataset/cnn-lstm-ctc/test.pkl'),
        is_shuffle = True,
        # is_shared = True,
        n = None,
        channels = 1):
    with open(file_path, 'r') as f:
        data = pkl.load(f)
        # chars = data['chars']
        # n_classes = len(chars)
        xs = data['x']
        ys = data['y']
        assert len(xs) == len(ys)
        n_samples = len(xs)
        height = xs[0].shape[0]
        x_max_len = np.max([x.shape[1] for x in xs])
        y_max_len = np.max([len(y) for y in ys])
        print("testing, x_max_step: {}, y_max_width: {}".format(x_max_len, y_max_len))

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
            y[i, :len(yy)] = yy
            y_clip[i] = len(yy)
        values = [x, x_mask, y, y_clip]

        # is shuffle
        if is_shuffle:
            perms = np.random.permutation(n_samples)
            values = [value[perms] for value in values]

        # slice
        if n is not None:
            n = min(n, n_samples)
            values = [value[:n] for value in values]

        values = [theano.shared(values[0]), theano.shared(values[1]), values[2], values[3]]
        # is shared
        # if is_shared:
            # values = [theano.shared(value) for value in values]
        return values

def compute_acc(y_pred, y_gt, y_clip_gt, chars):
    seqs_pred = []
    seqs_gt = []
    accs = []
    for i in range(len(y_pred)):
        shrink = []
        for j in range(len(y_pred[i])):
            if len(shrink) == 0 or shrink[-1] != y_pred[i, j]:
                shrink.append(y_pred[i, j])
        seq_pred = "".join([chars[c] for c in shrink if c != len(chars)])
        seq_gt = "".join([chars[y_gt[i, j]] for j in range(y_clip_gt[i])])
        acc = ed.eval(seq_pred, seq_gt) * 1.0 / len(seq_gt)
        seqs_pred.append(seq_pred)
        seqs_gt.append(seq_gt)
        accs.append(acc)
    return accs, seqs_pred, seqs_gt




