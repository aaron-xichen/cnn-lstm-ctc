#!/usr/bin/env python
# encoding: utf-8

import theano
import theano.tensor as T
import numpy as np
from utee import np_floatX, _p, shared
class LSTMLayer:
    # x: batcout_tmpsize x channel x height x n_steps
    # x_mask: n_samples, n_steps
    # to: n_steps x batcout_tmpsize x n_features
    # out: n_steps x batcout_tmpsize x n_hidden_units
    def __init__(self, x, x_mask, n_samples_const, n_features, n_hidden_units, prefix = 'lstm', forget=True):
        # shape check
        n_samples, n_steps = x_mask.shape
        self.x = x.reshape((n_samples, n_features, n_steps)).dimshuffle(2, 0, 1)
        self.x_mask = x_mask.T

        # init the parameters
        self.params = dict()
        num_activations = 3 + forget
        W = self.stacked_ortho_wts(n_features, n_hidden_units, num_activations)
        self.params[_p(prefix, 'W')] = shared(W)
        U = self.stacked_ortho_wts(n_hidden_units, n_hidden_units, num_activations)
        self.params[_p(prefix, 'U')] = shared(U)
        b = np.zeros(num_activations * n_hidden_units)
        self.params[_p(prefix, 'b')] = shared(b)

        out0 = shared(np.zeros((n_samples_const, n_hidden_units)).astype(theano.config.floatX))
        cell0 = shared(np.zeros((n_samples_const, n_hidden_units)).astype(theano.config.floatX))

        def _step(m_, x_, out_tmp, cell_tmp):
            preact = T.dot(out_tmp, self.params[_p(prefix, 'U')]) + x_

            inn_gate = T.nnet.sigmoid(preact[:, :n_hidden_units])
            out_gate = T.nnet.sigmoid(preact[:, n_hidden_units:2*n_hidden_units])
            fgt_gate = T.nnet.sigmoid(
                    preact[:, 2*n_hidden_units:3*n_hidden_units]) if forget else 1 - inn_gate

            # pre activation, tanh
            cell_val = T.tanh(preact[:, -n_hidden_units:])

            cell_val = fgt_gate * cell_tmp + inn_gate * cell_val
            cell_val = m_[:, None] * cell_val + (1. - m_)[:, None] * cell_tmp

            # after activation, linear
            out = out_gate * cell_val
            out = m_[:, None] * out + (1. - m_)[:, None] * out_tmp

            return out, cell_val

        self.param_w = [self.params[_p(prefix, 'W')]]
        self.param_b = [self.params[_p(prefix, 'b')]]
        self.x_prime = T.dot(self.x, self.params[_p(prefix, 'W')]) + self.params[_p(prefix, 'b')][None, None, :]

        rval, updates = theano.scan(_step,
                                    sequences=[self.x_mask, self.x_prime],
                                    outputs_info = [out0, cell0],
                                    name=_p(prefix, '_layers'))
        self.output = rval[0]
        self.n_out = n_hidden_units

    def ortho_weight(self, size):
        assert len(size) == 2
        m = np.max(size)
        W = np.random.randn(m, m).astype(theano.config.floatX)
        u, s, v = np.linalg.svd(W)
        return u[:size[0], :size[1]]

    def stacked_ortho_wts(self, n, m, copies):
        return np.hstack([self.ortho_weight((n, m)) for _ in range(copies)])


class BLSTMLayer:
    def __init__(self, x, x_mask, n_samples_const, n_features, n_hidden_units, forget=True):
        fwd = LSTMLayer(x, x_mask, n_samples_const, n_features, n_hidden_units, prefix='fw_lstm', forget=forget)
        bwd = LSTMLayer(x[:, :, :, ::-1], x_mask[:, ::-1], n_samples_const, n_features, n_hidden_units, prefix='bw_lstm', forget=forget)

        self.params = dict()
        self.params.update(fwd.params)
        self.params.update(bwd.params)

        self.n_out = fwd.n_out + bwd.n_out
        self.output =  T.concatenate([fwd.output, bwd.output[::-1]], axis=2)
        self.param_w = fwd.param_w + bwd.param_w # list merge
        self.param_b = fwd.param_b + bwd.param_b # list merge

