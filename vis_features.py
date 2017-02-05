#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from scipy.fftpack import dct, idct
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color, exposure, feature
from skimage.filters import roberts, sobel, scharr, prewitt, sobel_v, sobel_h


def elastic_transform(image, alpha, sigma, random_state=None):
    assert len(image.shape)==2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    return map_coordinates(image, indices, order=3, mode='nearest').reshape(shape)


data_path = './feat_constq/ey'
filename = './ey.interested'
ins_limit = 1
ds_limit = 1
alpha = 15
sigma = 2
rds = np.random.RandomState(1234)
dct = False
distort = True
use_edge = False
use_hog = False

filelist = [l for l in open(filename)] # all files in filename
interest_filelist = [''.join([ l.split()[0].split('.')[0], '.txt' ])
                        for l in filelist] # filenames of interest

plt.figure()
plt_idx = 1

for i in range(3):
    file = interest_filelist[i]
    feat = np.asarray([
        line.strip().split(',')
        for line in open(os.path.join(data_path, file))
    ], dtype = 'float')

    plt.subplot(3,6,plt_idx)
    plt_idx += 1
    plt.imshow(feat, extent=[0,0.5,0,0.5])
    plt.axis('off')

    if use_edge:
        edge_sobel = sobel_v(feat)
        plt.subplot(3,2,plt_idx)
        plt_idx += 1
        plt.imshow(edge_sobel, extent=[0,0.5,0,0.5])
        plt.axis('off')

    if use_hog:
        fd, hog_image = hog(feat, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(3, 3), visualise=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

        plt.subplot(2,2,plt_idx)
        plt_idx += 1
        plt.imshow(hog_image_rescaled, extent=[0,0.5,0,0.5])
        plt.axis('off')

    if dct:
        featvec_0 = dct(feat, axis=0, norm='ortho')
        plt.subplot(ins_limit,ds_limit+1,plt_idx)
        plt_idx += 1
        plt.imshow(featvec_0, extent=[0,0.5,0,0.5])

        featvec_1 = dct(feat, axis=1, norm='ortho')
        plt.subplot(ins_limit,ds_limit+1,plt_idx)
        plt_idx += 1
        plt.imshow(featvec_1, extent=[0,0.5,0,0.5])

        c_01 = dct(featvec_0, axis=1, norm='ortho')
        c_01[0:15, 0:15] = 0
        featvec_01 = idct(idct(c_01, axis=1), axis=0)
        featvec_01 = (featvec_01 - np.amin(featvec_01)) / \
            (np.amax(featvec_01) - np.amin(featvec_01))
        plt.subplot(ins_limit,ds_limit+1,plt_idx)
        plt_idx += 1
        plt.imshow(featvec_01, extent=[0,0.5,0,0.5])

        featvec_10 = dct(featvec_1, axis=0, norm='ortho')
        plt.subplot(ins_limit,ds_limit+1,plt_idx)
        plt_idx += 1
        plt.imshow(featvec_10, extent=[0,0.5,0,0.5])

    if distort:
        for j in range(5):
            featvec = elastic_transform(feat, alpha, sigma, random_state=rds)

            plt.subplot(3,6,plt_idx)
            plt_idx += 1
            plt.imshow(featvec, extent=[0,0.5,0,0.5])
            plt.axis('off')

plt.show()
# plt.savefig('./drafts/images/hog_feat.png')
