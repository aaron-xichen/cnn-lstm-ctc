#!/usr/bin/env python
# encoding: utf-8

import theano
import theano.tensor as T
# from theano import config
import numpy as np
from utee import np_floatX, _p, shared

class LSTMLayer:
    # x: batch_size x channel x height x n_steps
    # x_mask: n_samples, n_steps
    # to: n_steps x batch_size x n_features
    # out: n_steps x batch_size x n_hidden_units
    def __init__(self, x, x_mask, n_features, n_hidden_units, prefix = 'lstm'):
        # shape check
        n_samples, n_steps = x_mask.shape
        self.x = x.reshape((n_samples, n_features, n_steps)).dimshuffle(2, 0, 1)
        self.x_mask = x_mask.T

        # init the parameters
        self.params = dict()
        W = np.concatenate([self.guassian_weight((n_features, n_hidden_units)),
                            self.guassian_weight((n_features, n_hidden_units)),
                            self.guassian_weight((n_features, n_hidden_units)),
                            self.guassian_weight((n_features, n_hidden_units))],
                            axis=1)
        self.params[_p(prefix, 'W')] = shared(W)
        U = np.concatenate([self.ortho_weight(n_hidden_units),
                            self.ortho_weight(n_hidden_units),
                            self.ortho_weight(n_hidden_units),
                            self.ortho_weight(n_hidden_units)], axis=1)
        self.params[_p(prefix, 'U')] = shared(U)
        b = np.zeros((4 * n_hidden_units,))
        self.params[_p(prefix, 'b')] = shared(b)

        # build the lstm model
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(m_, x_, h_, c_):
            preact = T.dot(h_, self.params[_p(prefix, 'U')])
            preact += x_

            tmp = T.nnet.sigmoid(preact)
            i = _slice(tmp, 0, n_hidden_units)
            f = _slice(tmp, 1, n_hidden_units)
            o = _slice(tmp, 2, n_hidden_units)

            c = T.tanh(_slice(preact, 3, n_hidden_units))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * T.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

        x_prime = (T.dot(self.x, self.params[_p(prefix, 'W')]) +
                    self.params[_p(prefix, 'b')])

        rval, updates = theano.scan(_step,
                                    sequences=[self.x_mask, x_prime],
                                    outputs_info=[T.alloc(np_floatX(0.),
                                                            n_samples,
                                                            n_hidden_units),
                                                T.alloc(np_floatX(0.),
                                                            n_samples,
                                                            n_hidden_units)],
                                    name=_p(prefix, '_layers'),
                                    n_steps=n_steps)
        self.output = rval[0]

    def guassian_weight(self, size, mean = 0, std = 0.01):
        return np.random.normal(mean, std, size)

    def ortho_weight(self, ndim):
        W = np.random.randn(ndim, ndim)
        u, s, v = np.linalg.svd(W)
        return u
