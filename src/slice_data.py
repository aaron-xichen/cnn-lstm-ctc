#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import pickle as pkl

raw_file_path = os.path.expanduser('./data.pkl')
small_part_path = os.path.expanduser('./small.pkl')
medium_part_path = os.path.expanduser('./medium.pkl')
large_part_path = os.path.expanduser('./large.pkl')
test_part_path = os.path.expanduser('./test.pkl')

with open(raw_file_path, 'r') as f:
    print("load raw data")
    data = pkl.load(f)
    chars = data['chars']
    n_classes = len(chars)
    xs = data['x']
    ys = data['y']
    assert len(xs) == len(ys)
    n_samples = len(xs)
    perms = np.random.permutation(n_samples)
    xs_shuffle = [xs[idx] for idx in perms]
    ys_shuffle = [ys[idx] for idx in perms]

    # 10 percent for testing
    print("generate testing data")
    n_test = n_samples // 10
    x_test = [xs_shuffle[idx] for idx in range(n_test)]
    y_test = [ys_shuffle[idx] for idx in range(n_test)]
    data_test = {'x':x_test, 'y':y_test, 'chars':chars}
    with open(test_part_path, 'w') as fw:
        pkl.dump(data_test, fw)

    # remain training data
    xs = [xs_shuffle[idx] for idx in range(n_test, n_samples)]
    ys = [ys_shuffle[idx] for idx in range(n_test, n_samples)]
    n_samples = len(xs)


    # 1 percent of whole training data
    print("generating small training data")
    n_small = n_samples // 100
    x_small = [xs[idx] for idx in range(n_small)]
    y_small = [ys[idx] for idx in range(n_small)]
    data_small = {'x':x_small, 'y':y_small, 'chars':chars}
    with open(small_part_path, 'w') as fw:
        pkl.dump(data_small, fw)


    # 10 percent of whole training data
    print("generating medium training data")
    n_medium = n_samples // 10
    x_medium = [xs[idx] for idx in range(n_medium)]
    y_medium = [ys[idx] for idx in range(n_medium)]
    data_medium = {'x':x_medium, 'y':y_medium, 'chars':chars}
    with open(medium_part_path, 'w') as fw:
        pkl.dump(data_medium, fw)

    # 100 percent of whole training data
    print("generating whole training data")
    data_large = {'x':xs, 'y':ys, 'chars':chars}
    with open(large_part_path, 'w') as fw:
        pkl.dump(data_large, fw)
