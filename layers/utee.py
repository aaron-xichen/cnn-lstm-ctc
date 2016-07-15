#!/usr/bin/env python
# encoding: utf-8

import theano
from theano import config
import numpy as np
import pickle as pkl
import os
import editdistance as ed
import re
import cv2

def parse_bbox(im, text_path):
    height_std = 28.0
    chars = set([chr(x) for x in range(32, 127)])
    labels = []
    features = []
    with open(text_path, 'r')  as f:
        for line in f.readlines():
            try:
                candidates = re.findall('\[.+?\]', line)

                # process word
                word = candidates[1][1:-1]
                word = np.array([ord(c) for c in word])
                if np.max(word) > ord(chars[-1]) or np.min(word) < ord(chars[0]):
                    continue
                word = word - ord(chars[0]);

                # process cords
                cords = candidates[2][1:-1].split(',')
                up = int(cords[0])
                down = int(cords[1]) + 1
                left = int(cords[2])
                right = int(cords[3]) + 1
                sub = im[up:down, left:right]
                new_width = int(height_std / sub.shape[0] * sub.shape[1])
                sub = cv2.resize(sub, (new_width, int(height_std)))

                features.append(sub)
                labels.append(word.tolist())
            except Exception as e:
                print(e)
                print("parse wrong, skip")
    return features, labels

def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def shared(data, name = None):
    if name is not None:
        return theano.shared(np_floatX(data), name = name)
    else:
        return theano.shared(np_floatX(data))

def _p(pp, name):
    return '{}_{}'.format(pp, name)

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

class Prefetcher():
    def __init__(self, img_list_path, imgs_dir, batch_size,
            stride = 1, patch_width = [1], offset = None, is_shared = False):
        self.imgs_dirs = imgs_dir
        self.img_list = []
        self.labels = []
        with open(img_list_path, 'rb') as f:
            l = f.readlines()
        print("loaded {} samples from {}".format(len(l) - 1, img_list_path))
        chars_from, chars_to = l[0].split(' ')
        self.chars = [chr(x) for x in range(int(chars_from), int(chars_to))]
        for r in l[1:]:
            fields = r.strip().split(' ')
            self.img_list.append(fields[0])
            self.labels.append([int(c) for c in fields[1:]])
        self.batch_size = batch_size
        self.n_samples = len(self.img_list)
        # self.idxs = np.random.permutation(self.n_samples)
        self.idxs = range(self.n_samples)
        if offset is not None:
            self.cur = offset % self.n_samples
        else:
            self.cur = 0
        self.stride = stride
        self.patch_width = patch_width
        self.n_classes = len(self.chars)
        self.is_shared = is_shared


    def fetch_next(self, is_blank_y = True):
        features = []
        labels = []
        # load into memory
        while len(features) < self.batch_size:
            if self.cur >= self.n_samples:
                self.cur = 0
                self.idxs = np.random.permutation(self.n_samples)
            img_path = self.img_list[self.idxs[self.cur]]
            full_img_path = os.path.join(self.imgs_dirs, img_path)
            im = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
            features.append(im / 255.)
            labels.append(self.labels[self.idxs[self.cur]])
            self.cur += 1
        values = self._wrap(features, labels, self.batch_size, self.stride, self.patch_width, self.n_classes, self.is_shared, is_blank_y)
        return values

    def _wrap(self, features, labels, batch_size, stride, patch_width, n_classes, is_shared, is_blank_y):
        # packing
        x_max_len = np.max([x.shape[1] for x in features])
        y_max_len = 50 # pre-difine
        height = features[0].shape[0]

        # transform
        x_max_len = np.ceil(x_max_len * 1. / stride)
        height = height * np.sum(patch_width)
        print("[prefetch]height: {}, x_max_step:{}, y_max_width:{}".format(height, x_max_len, y_max_len))

        # x and x_mask
        x = np.zeros((batch_size, 1, height, x_max_len)). astype(config.floatX)
        x_mask = np.zeros((batch_size, x_max_len)).astype(config.floatX)
        for i, xx in enumerate(features):
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
        y = np.zeros((batch_size, y_max_len)).astype('int32')
        y_clip = np.zeros((batch_size)).astype('int32')
        if is_blank_y:
            for i, yy in enumerate(labels):
                y_extend = np.ones(2 * len(yy) + 1, dtype='int32') * n_classes
                for j in range(len(yy)):
                    y_extend[2 * j + 1] = yy[j]
                y[i, :len(y_extend)] = y_extend
                y_clip[i] = len(y_extend)
        else:
            for i, yy in enumerate(labels):
                y[i, :len(yy)] = yy;
                y_clip[i] = len(yy)

        values = [x, x_mask, y, y_clip]

        # is shared
        if is_shared:
            values = [theano.shared(value) for value in values]

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
