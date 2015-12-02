#!/usr/bin/env python
# encoding: utf-8
import theano
import theano.tensor as T
eps, epsinv = 1e-30, 1e30

# def safe_log(x):
    # return T.log(T.maximum(x, eps).astype(theano.config.floatX))

# def safe_exp(x):
    # return T.exp(T.minimum(x, epsinv).astype(theano.config.floatX))

# def logadd_simple(x, y):
    # return x + safe_log(1 + safe_exp(y - x))

# def logadd_a# dvanced(x, y):
    # maxx = tt.maximum(x, y)
    # minn = tt.minimum(x, y)
    # return maxx + tt.log(1 + tt.exp(minn - maxx))

# def logadd(x, y, *zs, add=logadd_simple):
    # sum = add(x, y)
    # for z in zs:
        # sum = add(sum, z)
    # return sum

# def logmul(x, y):
    # return x + y

class CTCLayer():
    def __init__(self, x, x_mask, y, y_clip,
            blank, prefix = 'ctc', log_space = True):
        self.x = x
        self.x_mask = x_mask
        self.y = y
        self.y_clip = y_clip
        self.prefix = prefix
        self.blank = blank
        if log_space:
            self.log_ctc()
        else:
            self.plain_ctc()

    def plain_ctc(self):
        # prepare y
        n_samples, labels_len = self.y.shape
        y1 = T.concatenate([self.y, T.ones((self.y.shape[0], 2)) * self.blank], axis=1)
        diag = T.neq(y1[:, :-2], y1[:, 2:]) * T.neq(y1[:, :-2], self.blank)
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
                outputs_info=T.set_subtensor(T.zeros((n_samples, labels_len), dtype=theano.config.floatX)[:, 0], 1),
                non_sequences = [diags]
                )

        # prepare y_clip
        y_clip1 = T.concatenate([(self.y_clip - 1)[:, None], self.y_clip[:, None]], axis = 1)
        self.prob = self.debug[-1][T.arange(n_samples)[:, None], y_clip1]
        self.pin = -T.log(T.maximum(T.sum(self.prob, axis=1), eps))
        # self.loss = T.mean(-T.log(T.sum(self.prob, axis=1)))
        self.loss = T.mean(self.pin)

    def log_ctc(self):
        print("TODO")

