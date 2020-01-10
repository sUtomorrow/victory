# -*- coding: utf-8 -*-
# @Time     : 7/1/19 9:35 PM
# @Author   : lty
# @File     : img_show

import os
import numpy as np
import cv2
import pickle

img_path = '/mnt/data4/lty/victory_data/testB_dataset_spacing662/647177.npy'
pkl_path = img_path.replace('.npy', '.pkl')

img_array = np.load(img_path)
print(img_array.shape)
img_array = (np.clip(img_array, -1000, 400) + 1000) / 1400

with open(pkl_path, 'rb') as fp:
    img_info = pickle.load(fp)
print(img_info)
for img in img_array:
    cv2.imshow('img', img)
    cv2.waitKey()