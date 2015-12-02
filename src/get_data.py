#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import pickle as pkl
import cv2
import os
import re
import sys

# images_dir = os.path.expanduser('~/Documents/dataset/boxed_data/images')
# boxes_dir = os.path.expanduser('~/Documents/dataset/boxed_data/trans')
# save_path = os.path.expanduser('~/Documents/dataset/boxed_data/data.pkl')

height_std = 28.0
features_all = []
labels_all = []
chars = [chr(x) for x in range(33, 127)]

def parse_bbox(im, text_path):
    labels = []
    features = []
    with open(text_path, 'r')  as f:
        for line in f.readlines():
            try:
                candidates = re.findall('\[.+?\]', line)

                # process word
                word = candidates[1][1:-1]
                word = np.array([ord(c) for c in word])
                if np.max(word) > ord(chars[-1]) or np.min(word) < ord(chars[0]):
                    continue
                word = word - ord(chars[0]);

                # process cords
                cords = candidates[2][1:-1].split(',')
                up = int(cords[0])
                down = int(cords[1]) + 1
                left = int(cords[2])
                right = int(cords[3]) + 1
                sub = im[up:down, left:right]
                new_width = int(height_std / sub.shape[0] * sub.shape[1])
                sub = cv2.resize(sub, (new_width, int(height_std)))

                features.append(sub)
                labels.append(word.tolist())
            except Exception as e:
                print(e)
                # print("im shape: {}, sub shape: {}".format(im.shape, sub.shape))
                # print("up: {}, down:{}, left: {}, right: {}".format(up, down, left, right))
                print("parse wrong, skip")
    return features, labels

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 3:
        print("USAGE: get_data.py img_dir box_dir out_dir")
        sys.exit(1)

    images_dir = args[0]
    boxes_dir = args[1]
    save_path = args[2]

    for root, dirs, files in os.walk(images_dir):
        for i, file in enumerate(files):
            if file.endswith('.jpg'):
                print("processing {}({}/{})".format(file, i + 1, len(files)))
                image_full_path = os.path.join(root, file)
                box_full_path = os.path.join(boxes_dir, file[0:-4] + '.box')
                im = cv2.imread(image_full_path);
                im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float) / 255.0;
                features, labels = parse_bbox(im_grey, box_full_path)
                labels_all.extend(labels)
                features_all.extend(features)

    print("total samples: {}".format(len(labels_all)))

    with open(save_path, 'wb') as f:
        dt = {'x': features_all, 'y': labels_all, 'chars': chars};
        pkl.dump(dt, f, -1)
