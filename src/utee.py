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

# make a snapshot, params must be shared variable
def snapshot(file_path, net):
    print("saving snapshot to {}".format(file_path))
    cellar = dict()
    assert isinstance(net.params, dict)
    for key, value in net.params.items():
        cellar[key] = value.get_value()
    with open(file_path, 'wb') as f:
        pkl.dump(cellar, f)

# resume from snaphot, params should be shared variable
def resume_model(file_path, net):
    print("resuming snapshot from {}".format(file_path))
    with open(file_path, 'rb') as f:
        params = pkl.load(f)
    assert isinstance(net.params, dict)
    for key, value in net.params.items():
        assert key in params
        value.set_value(params[key].astype(config.floatX))

def prepare_training_data(
        file_path = None,
        is_shuffle = True,
        is_shared = True,
        n = None,
        channels = 1,
        stride = 1,
        patch_width = [1]):
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
        # transform
        x_max_len = np.ceil(x_max_len * 1. / stride)
        height = height * np.sum(patch_width)
        print("training data, height: {}, x_max_step:{}, y_max_width:{}, n_samples:{}".
                format(height, x_max_len, y_max_len, n_samples))

        # x and x_mask
        x = np.zeros((n_samples, channels, height, x_max_len)). astype(config.floatX)
        x_mask = np.zeros((n_samples, x_max_len)).astype(config.floatX)
        for i, xx in enumerate(xs):
            shape = xx.shape
            l = int(np.ceil(xx.shape[1] * 1. / stride))
            for j in range(l):
                long_vec = []
                base = j * stride
                for patch in patch_width:
                    vec = np.zeros(shape[0] * patch).astype(config.floatX)
                    vec2 = xx[:, base:base+patch].T.flatten()
                    vec[:len(vec2)] = vec2
                    long_vec = np.concatenate([long_vec, vec])
                assert len(long_vec) == height
                x[i, :, :, j] = long_vec
            x_mask[i, :l] = 1.0


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
        n = None,
        channels = 1,
        stride = 1,
        patch_width = [1]
        ):
    with open(file_path, 'r') as f:
        data = pkl.load(f)
        xs = data['x']
        ys = data['y']
        chars = data['chars']
        assert len(xs) == len(ys)
        n_samples = len(xs)
        height = xs[0].shape[0]
        x_max_len = np.max([x.shape[1] for x in xs])
        y_max_len = np.max([len(y) for y in ys])
        # transform
        x_max_len = np.ceil(x_max_len * 1. / stride)
        height = height * np.sum(patch_width)
        print("testing data, height: {}, x_max_step:{}, y_max_width:{}, n_samples:{}".
                format(height, x_max_len, y_max_len, n_samples))

        # x and x_mask
        x = np.zeros((n_samples, channels, height, x_max_len)). astype(config.floatX)
        x_mask = np.zeros((n_samples, x_max_len)).astype(config.floatX)
        for i, xx in enumerate(xs):
            shape = xx.shape
            l = int(np.ceil(xx.shape[1] * 1. / stride))
            for j in range(l):
                long_vec = []
                base = j * stride
                for patch in patch_width:
                    vec = np.zeros(shape[0] * patch).astype(config.floatX)
                    vec2 = xx[:, base:base+patch].T.flatten()
                    vec[:len(vec2)] = vec2
                    long_vec = np.concatenate([long_vec, vec])
                assert len(long_vec) == height
                x[i, :, :, j] = long_vec
            x_mask[i, :l] = 1.0

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
        values.append(chars)
        values.append(height)
        return values

def compute_acc(y_pred, y_gt, y_clip_gt, chars):
    seqs_pred = []
    seqs_gt = []
    accs = []
    values = []
    for i in range(len(y_pred)):
        shrink = []
        for j in range(len(y_pred[i])):
            if len(shrink) == 0 or shrink[-1] != y_pred[i, j]:
                shrink.append(y_pred[i, j])
        seq_pred = "".join([chars[c] for c in shrink if c != len(chars)])
        seq_gt = "".join([chars[y_gt[i, j]] for j in range(y_clip_gt[i])])
        value = ed.eval(seq_pred, seq_gt) * 1.0 / len(seq_gt)
        acc = seq_pred == seq_gt
        seqs_pred.append(seq_pred)
        seqs_gt.append(seq_gt)
        values.append(value)
        accs.append(acc)
    return seqs_pred, seqs_gt, accs, values




