#!/usr/bin/env python
# encoding: utf-8
import theano
import theano.tensor as T
import numpy as np

eps, epsinv = 1e-40, 1e40
def safe_log(x):
    return T.log(T.maximum(x, eps).astype(theano.config.floatX))

def safe_exp(x):
    return T.exp(T.minimum(x, epsinv).astype(theano.config.floatX))

class CTCLayer():
    def __init__(self, x, x_mask, y, y_clip,
            labels_len_const,
            blank, prefix = 'ctc', log_space = True):
        self.x = x
        self.x_mask = x_mask
        self.y = y
        self.y_clip = y_clip
        self.prefix = prefix
        self.blank = blank
        if log_space:
            self.log_ctc(labels_len_const = labels_len_const)
        else:
            self.plain_ctc()

    def plain_ctc(self):
        def _build_diag(_d):
            extend_I = T.eye(labels_len + 2)
            return T.eye(labels_len) + extend_I[1:-1, :-2] + extend_I[2:, :-2] * _d[:, None]

        # prepare y
        n_samples, labels_len = self.y.shape
        y1 = T.concatenate([self.y, T.ones((self.y.shape[0], 2)) * self.blank], axis=1)
        diag = T.neq(y1[:, :-2], y1[:, 2:]) * T.neq(y1[:, 2:], self.blank)
        # stretch out, (labels_len, n_samples*labels_len)
        diags0, _ = theano.scan(fn = _build_diag,
                sequences = [diag],
                n_steps = n_samples)
        shape = diags0.shape
        diags = T.transpose(diags0, (1, 0, 2)).reshape((shape[1], shape[0] * shape[2]))

        # prepare x
        assert self.x.ndim == 3
        # (n_steps, n_samples, softmax_output) to (n_steps, n_samples, labels_len)
        x1 = self.x[:, T.arange(n_samples)[:, None], self.y]
        dims = x1.shape
        # stretch out, (n_steps, n_samples * labels_len)
        x2 = x1.reshape((dims[0], dims[1] * dims[2]))

        # each step
        def _step(m_, s_, h_, diags):
            tmp1 = T.dot(h_, diags) * s_[None, :]
            tmp2 = tmp1.reshape((n_samples, n_samples, labels_len))
            slic = tmp2[T.arange(n_samples)[:, None], T.arange(n_samples)[:, None], :]
            slic = slic.reshape((slic.shape[0], slic.shape[2]))
            p = m_[:, None] * slic + (1 - m_)[:, None] * h_
            return p

        # scan loop
        self.debug, _ = theano.scan(fn = _step,
                sequences = [self.x_mask.T, x2],
                outputs_info = [T.set_subtensor(T.zeros((n_samples, labels_len), dtype=theano.config.floatX)[:, 0], 1)],
                non_sequences = [diags]
                )

        # prepare y_clip
        y_clip1 = T.concatenate([(self.y_clip - 2)[:, None], (self.y_clip - 1)[:, None]], axis = 1)
        self.prob = self.debug[-1][T.arange(n_samples)[:, None], y_clip1]
        self.loss = T.mean(-T.log(T.sum(self.prob, axis=1)))

    def log_ctc(self, labels_len_const):
        def _build_diag(_d):
            extend_I = T.eye(labels_len + 2)
            return T.eye(labels_len) + extend_I[1:-1, :-2] + extend_I[2:, :-2] * _d[:, None]

        # prepare y
        n_samples, labels_len = self.y.shape
        y1 = T.concatenate([self.y, T.ones((self.y.shape[0], 2)) * self.blank], axis=1)
        diag = T.neq(y1[:, :-2], y1[:, 2:]) * T.neq(y1[:, 2:], self.blank)
        # stretch out, (labels_len, n_samples*labels_len)
        diags0, _ = theano.scan(fn = _build_diag,
                sequences = [diag],
                n_steps = n_samples)
        shape = diags0.shape
        diags = T.transpose(diags0, (1, 0, 2)).reshape((shape[1], shape[0] * shape[2]))

        # prepare x
        assert self.x.ndim == 3
        # (n_steps, n_samples, softmax_output) to (n_steps, n_samples, labels_len)
        x1 = self.x[:, T.arange(n_samples)[:, None], self.y]
        dims = x1.shape
        # stretch out, (n_steps, n_samples * labels_len)
        x2 = x1.reshape((dims[0], dims[1] * dims[2]))

        def log_matrix_dot(x, y, z):
            v1 = x[:, :, None]
            v2 = T.tile(v1, (1, 1, labels_len_const))
            v2_shape = v2.shape
            v3 = T.transpose(v2, (1, 0, 2)).reshape((v2_shape[1], v2_shape[0] * v2_shape[2]))
            v4 = v3 + y
            m = T.max(v4, axis=0)

            v5 = v4 - m[None, :]
            # mask = T.nonzero(T.isnan(v5))
            # v6 = T.set_subtensor(v5[mask], -np.inf)
            # v7 = T.exp(v5)
            v7 = safe_exp(v5)
            v8 = T.sum(v7, axis=0)
            # v9 = T.log(v8)
            v9 = safe_log(v8)
            v10 = v9 + m
            v11 = v10 + z
            v12 = v11.reshape((n_samples, labels_len))
            return v12

        # each step
        def _step(m_, s_, h_, diags):
            v = log_matrix_dot(h_, diags, s_)
            m_extend = T.tile(m_[:, None], (1, labels_len_const))
            p = T.switch(m_extend, v, h_)
            return p

        # scan loop
        log_x2 = safe_log(x2)
        log_outputs_info = safe_log(T.set_subtensor(T.zeros((n_samples, labels_len), dtype=theano.config.floatX)[:, 0], 1))
        log_diags = safe_log(diags)

        self.pin0 = log_x2
        self.pin1 = log_outputs_info
        self.pin2 = log_diags

        self.debug, _ = theano.scan(fn = _step,
                sequences = [self.x_mask.T, log_x2],
                outputs_info = [log_outputs_info],
                non_sequences = [log_diags]
                )

        # prepare y_clip
        y_clip1 = T.concatenate([(self.y_clip - 2)[:, None], (self.y_clip - 1)[:, None]], axis = 1)
        self.prob = self.debug[-1][T.arange(n_samples)[:, None], y_clip1]

        # compute loss
        mx = T.max(self.prob, axis=1)
        l1 = self.prob - mx[:, None]
        # l2 = T.sum(T.exp(l1), axis=1)
        # l3 = T.log(l2) + mx
        l2 = T.sum(safe_exp(l1), axis=1)
        l3 = safe_log(l2) + mx
        self.loss = T.mean(-l3)
