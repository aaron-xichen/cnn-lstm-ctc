#!/usr/bin/env python
# encoding: utf-8
import theano
from theano import config
import numpy as np

def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def shared(data, name = None):
    if name is not None:
        return theano.shared(np_floatX(data), name = name)
    else:
        return theano.shared(np_floatX(data))

def _p(pp, name):
    return '%s_%s' % (pp, name)

