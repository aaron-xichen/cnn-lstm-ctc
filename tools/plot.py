#!/usr/bin/env python
# encoding: utf-8
import pickle as pkl
import matplotlib.pyplot as plt

with open('info.pkl', 'r') as f:
    weight = pkl.load(f)

for key, value in weight.iteritems():
    plt.plot(value, label=key)
plt.legend()

plt.show()


