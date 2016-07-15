#!/usr/bin/env python
# encoding: utf-8

import theano
import theano.tensor as T
import numpy as np
import time
from layers.net import Net
from layers.lstm_layer import BLSTMLayer
from layers.utee import resume_model

class Recognition():
    def _build_network(self, tic = None):
        print("building network begin({})".format(time.time() - tic))
        assert tic is not None
        # build symbolic expression
        print("building symbolic tensors({})".format(time.time() - tic))
        x =  T.tensor4('x')
        x_mask = T.matrix('x_mask')
        y = T.imatrix('y') # useless, only for API
        y_clip = T.ivector('y_clip') # useless, only for API

        # setting parameters
        print("setting parameters({})".format(time.time() - tic))
        lstm_hidden_units = 90
        self.height_raw = 28 # original fixed height of image
        self.height = self.height_raw * np.sum(self.patch_width)
        n_classes = len(self.chars)

        # network configuration
        options = dict()
        options['n_in_lstm_layer'] = self.height
        options['n_out_lstm_layer'] = lstm_hidden_units
        options['n_out_hidden_layer'] = n_classes + 1 # additional class blank
        options['blank'] = n_classes
        options['labels_len'] = 50
        options['batch_size'] = self.batch_size
        options['n_classes'] = n_classes
        print(options)

        # build predict model
        print("building the predict model({})".format(time.time() - tic))
        self.net = Net(x = x, x_mask = x_mask, y = y, y_clip = y_clip, options = options,
                mid_layer_type = BLSTMLayer, forget=False)

        # build predict function
        print("building the predict function({})".format(time.time() - tic))
        self.predict = theano.function(
                inputs = [x, x_mask],
                outputs = [self.net.pred, self.net.softmax_matrix],
                )
        print("building network done({})".format(time.time() - tic))

    def _load_model(self, model_path = None, tic = None):
        print("loading model begin({})".format(time.time() - tic))
        assert model_path is not None
        assert tic is not None
        print("loading model from {}".format(model_path))
        resume_model(model_path, self.net)
        print("loading model done({})".format(time.time() - tic))

    def _pack_imgs(self, imgs, tic = None):
        print("packing images begin({})".format(time.time() - tic))
        assert tic is not None

        # compute the x_max_len and y_max_len
        x_max_len = 0
        for img in imgs:
            assert len(img) == self.height_raw
            x_max_len = max(len(img[0]), x_max_len)
        y_max_len = 50 # pre-difine
        x_max_len = np.ceil(x_max_len * 1. / self.stride)
        print("[pack image]height: {}, x_max_step:{}, y_max_width:{}".format(self.height, x_max_len, y_max_len))

        # packing
        imgs_n_samples = len(imgs)
        x = np.zeros((imgs_n_samples, 1, self.height, x_max_len)). astype(theano.config.floatX)
        x_mask = np.zeros((imgs_n_samples, x_max_len)).astype(theano.config.floatX)
        for i, img in enumerate(imgs):
            xx = np.array(img)
            shape = xx.shape
            l = int(np.ceil(xx.shape[1] * 1. / self.stride))
            for j in range(l):
                long_vec = []
                base = j * self.stride
                for patch in self.patch_width:
                    vec = np.zeros(shape[0] * patch).astype(theano.config.floatX)
                    vec2 = xx[:, base:base+patch].T.flatten()
                    vec[:len(vec2)] = vec2
                    long_vec = np.concatenate([long_vec, vec])
                assert len(long_vec) == self.height
                x[i, :, :, j] = long_vec
            x_mask[i, :l] = 1.0
        print("packing images done({})".format(time.time() - tic))
        return x, x_mask

    def _compute_confidence(self, pred, softmax_matrix):
        seq_new = []
        confidence_new = []
        for i in range(len(pred)):
            shrink = []
            confidence = []
            count = []
            for j in range(len(pred[i])):
                if len(shrink) == 0 or shrink[-1] != pred[i, j]:
                    shrink.append(pred[i, j])
                    confidence.append(softmax_matrix[j, i, pred[i, j]])
                    count.append(1)
                else:
                    confidence[-1] = (confidence[-1] * count[-1] + softmax_matrix[j, i, pred[i, j]]) / (count[-1] + 1)
                    count[-1] += 1
            shrink_final = []
            confidence_final = []
            for j, c in enumerate(shrink):
                if c != len(self.chars):
                    shrink_final.append(self.chars[c])
                    confidence_final.append(confidence[j])
            seq_new.append("".join(shrink_final))
            confidence_new.append(confidence_final)
            assert len(seq_new[-1]) == len(confidence_new[-1])
        return seq_new, confidence_new

    def recog(self, imgs):
        tic = time.time()
        imgs_n_samples = len(imgs)
        print("received new {} images({})".format(imgs_n_samples, time.time() - tic))
        n_batch = int(np.ceil(imgs_n_samples * 1.0 / self.batch_size))
        print("split to {} batch({})".format(n_batch, time.time() - tic))

        # packing image first
        packed_imgs, packed_imgs_mask = self._pack_imgs(imgs, tic)
        packed_imgs_shape = packed_imgs.shape
        packed_imgs_mask_shape = packed_imgs_mask.shape
        assert imgs_n_samples == len(packed_imgs)
        assert imgs_n_samples == len(packed_imgs_mask)

        # recognizing
        seqs = []
        confidences = []
        for i in range(n_batch):
            x = np.zeros((self.batch_size, packed_imgs_shape[1], packed_imgs_shape[2], packed_imgs_shape[3])).astype(theano.config.floatX)
            x_mask = np.zeros((self.batch_size, packed_imgs_mask_shape[1])).astype(theano.config.floatX)
            begin = i * self.batch_size
            end = min((i + 1) * self.batch_size, imgs_n_samples)
            x[:(end-begin), :, :, :] = packed_imgs[begin:end, :, :, :]
            x_mask[:(end-begin), :] = packed_imgs_mask[begin:end, :]
            print("processing batch {}({})".format(i + 1, time.time() - tic))
            pred, softmax_matrix = self.predict(x, x_mask)
            seq_new, confidence_new = self._compute_confidence(pred, softmax_matrix)
            seqs.extend(seq_new)
            confidences.extend(confidence_new)
        recog_result = zip(seqs, confidences)
        print("send back result({})".format(time.time() - tic))
        return recog_result[:imgs_n_samples]

    def init(self, model_path, batch_size = 64, stride = 1, patch_width = [1]):
        print("Init Recognition Module")
        tic = time.time()
        self.batch_size = batch_size
        self.stride = stride
        self.patch_width = patch_width
        self.chars = [chr(c) for c in range(32, 127)]

        # build network and model
        self._build_network(tic)
        self._load_model(model_path, tic)
        print("Init Recognition Module Done({})".format(time.time() - tic))
