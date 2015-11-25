#!/usr/bin/env python
# encoding: utf-8
import theano
import theano.tensor as T
class CTCLayer():
    def __init__(self, x, x_mask, y, y_clip,
            blank, prefix = 'ctc'):
        # prepare y
        n_samples, labels_len = y.shape
        y1 = T.concatenate([y, T.ones((y.shape[0], 2)) * blank], axis=1)
        diag = T.neq(y1[:, :-2], y1[:, 2:]) * T.neq(y1[:, :-2], blank)
        def _build_diag(_d):
            value = T.eye(labels_len) + \
            T.eye(labels_len, k=1) + \
            T.eye(labels_len, k=2) * _d.dimshuffle((0, 'x'))
            return value
        # stretch out, (labels_len, n_samples*labels_len)
        diags0, _ = theano.scan(fn = _build_diag,
                sequences = [diag],
                n_steps = n_samples)
        shape = diags0.shape
        diags = T.transpose(diags0, (1, 0, 2)).reshape((shape[1], shape[0] * shape[2]))

        # prepare x
        print(x.ndim)
        assert x.ndim == 3
        # (n_steps, n_samples, softmax_output) to (n_steps, n_samples, labels_len)
        x1 = x[:, T.arange(n_samples)[:, None], y]
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
                sequences = [x_mask.T, x2],
                outputs_info=T.set_subtensor(T.zeros((n_samples, labels_len), dtype=theano.config.floatX)[:, 0], 1),
                non_sequences = [diags]
                )

        # prepare y_clip
        y_clip1 = T.concatenate([(y_clip - 1)[:, None], y_clip[:, None]], axis = 1)
        self.prob = self.debug[-1][T.arange(n_samples)[:, None], y_clip1]
        self.loss = T.mean(-T.log(T.sum(self.prob, axis=1)))
