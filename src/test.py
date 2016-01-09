#!/usr/bin/env python
# encoding: utf-8

from threading import Thread, Lock, Condition
import numpy as np
import theano

shared_var = theano.shared(np.zeros(5,5).astype(theano.config.floatX))
lock = Lock()

class prefetcher(Thread):
    def __int__(self):
        self.is_stop = False
    def run(self):
        while not self.is_stop:
            con.acquire()
            print("prefetcher get it")
            shared_var.set_value(np.random.rand(5,5).astype(theano.config.floatX))
            con.release()
            con.wait()
    def stop(self):
        self.is_stop = True


x = theano.marix('x')





