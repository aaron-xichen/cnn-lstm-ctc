#!/usr/bin/env python
# encoding: utf-8

import recog_module
import numpy as np
import cv2

reg = recog_module.Recognition()
reg.init("99.pkl")

imgs = []
for i in range(100):
    im = cv2.imread('/share/data_for_BLSTM_CTC/Samples_for_English/20151205/imgs/{}.jpg'.format(i + 1))
    im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    imgs.append(im_grey)

result = reg.recog(imgs)
print("out: ", len(result))
for i, img in enumerate(imgs):
    save_path = 'out/' + result[i][0] + '.jpg'
    cv2.imwrite(save_path, img * 255)
    print("predict result: " +  result[i][0])
