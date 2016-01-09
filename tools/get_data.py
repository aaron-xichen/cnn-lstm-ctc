#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import cv2
import os
import re

height_std = 28.0
chars = [chr(x) for x in range(32, 127)]

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
                print("parse wrong, skip")
    return features, labels

if __name__ == "__main__":
    images_dir = os.path.expanduser('./images/')
    boxes_dir = os.path.expanduser("./trans/")
    out_imgs_dir = os.path.expanduser('./imgs/')
    out_train_imgs_list_path = os.path.expanduser('./train_img_list.txt')
    out_test_imgs_list_path = os.path.expanduser('./test_img_list.txt')

    features_all = []
    labels_all = []
    for root, dirs, files in os.walk(images_dir):
        for i, file in enumerate(files):
            if file.endswith('.jpg'):
                print("processing {}({}/{})".format(file, i + 1, len(files)))
                image_full_path = os.path.join(root, file)
                box_full_path = os.path.join(boxes_dir, file[0:-4] + '.box')
                im = cv2.imread(image_full_path);
                im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float)
                features, labels = parse_bbox(im_grey, box_full_path)
                labels_all.extend(labels)
                features_all.extend(features)

    print("total samples: {}".format(len(labels_all)))
    idxs = np.random.permutation(len(features_all))
    n_train_samples = int(0.9 * len(features_all))
    print("n_train_samples: {}, n_test_samples{}".format(n_train_samples, len(features_all) - n_train_samples))

    # for training
    print("saving to {}".format(out_train_imgs_list_path))
    with open(out_train_imgs_list_path, 'wb') as f:
        f.write("32 127\n")
        for i in idxs[:n_train_samples]:
            idx = idxs[i]
            img_save_path = "{}.jpg".format(i)
            img_save_full_path = os.path.join(out_imgs_dir, img_save_path)
            cv2.imwrite(img_save_full_path, features_all[idx])
            labels_str = [str(c) for c in labels_all[idx]]
            record = img_save_path + " " + " ".join(labels_str) + "\n"
            f.write(record)

    # for testing
    print("saving to {}".format(out_test_imgs_list_path))
    with open(out_test_imgs_list_path, 'wb') as f:
        f.write("32 127\n")
        for i in idxs[n_train_samples:]:
            idx = idxs[i]
            img_save_path = "{}.jpg".format(i)
            img_save_full_path = os.path.join(out_imgs_dir, img_save_path)
            cv2.imwrite(img_save_full_path, features_all[idx])
            labels_str = [str(c) for c in labels_all[idx]]
            record = img_save_path + " " + " ".join(labels_str) + "\n"
            f.write(record)
