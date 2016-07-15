#!/usr/bin/env python
# encoding: utf-8

import sys
assert len(sys.argv) == 4, "Usage: python predict.py model_path test_imgs_list_path output_root" 

import recog_module
import numpy as np
import cv2
import os

model_path = sys.argv[1]
test_imgs_list_path = sys.argv[2]
save_root = sys.argv[3]

# init recog_module from model_path
reg = recog_module.Recognition()
reg.init(model_path)

# read test_imgs_list_path
dir_name = os.path.dirname(test_imgs_list_path)
with open(test_imgs_list_path, 'rb') as f:
    l = f.readlines()

# loaded images from disk
imgs = []
print("loaded {} samples from {}".format(len(l)-1, test_imgs_list_path))
for r in l[1:]:
    fields = r.strip().split(' ')
    full_path = os.path.join(dir_name, 'split_tiny_images', fields[0])
    im = cv2.imread(full_path)
    im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    imgs.append(im_grey)

# recog all, automatically dispatch into batches
result = reg.recog(imgs)
print("out: ", len(result))
for i, img in enumerate(imgs):
    save_path = os.path.join(save_root, str(i) + '-' + result[i][0] + '.jpg')
    cv2.imwrite(save_path, img * 255)
    print("predict result: " +  result[i][0])
