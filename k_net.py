#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import sys
import os

import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense, Convolution2D, \
    MaxPooling2D, AveragePooling1D, ZeroPadding2D, Dropout, \
    Flatten, LSTM, Input, Reshape
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

"""
Keras model related.
"""
class NetBuilder:
    """
    Net builder with Keras.
    """
    def __init__(self, inp, lbl):
        nrow, ncol = inp.shape
        nout = lbl.shape[0]
        inp = inp.reshape((1, 1, nrow, ncol))
        lbl = lbl.reshape((1, nout))
        self.input = inp
        self.output = lbl

        self.model = Sequential()
        self.model.add(Convolution2D(
            10, nrow, 1,
            input_shape = (1, nrow, ncol),
            subsample = (1, 1)
        )
                       )
        self.model.add(Activation('relu'))
        self.model.add(Reshape((10, ncol)))
        self.model.add(LSTM(100))
        #self.model.add(AveragePooling1D(pool_length=2))
        self.model.add(Dense(nout))
        self.model.add(Activation('softmax'))

    def trainer(self, nepoch):
        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy']
        )
        if DEBUG_OUTPUT:
            print(self.model.layers[0].output_shape)
            print(self.model.layers[1].output_shape)
            print(self.model.layers[2].output_shape)
            print(self.model.layers[3].output_shape)
            print(self.model.layers[4].output_shape)
            print(self.model.layers[5].output_shape)

        hist = self.model.fit(
            self.input, self.output,
            nb_epoch = nepoch
        )

    def tester(self, x, y):
        nr, nc = x.shape
        no = y.shape[0]
        x = x.reshape((1, 1, nr, nc))
        y = y.reshape((1, no))
        eval = self.model.evaluate(x, y)

    def saver(self):
        pass

def run_model():
    dpath = './feat_constq'
    read_instance(os.path.join(dpath ,'ey'), './ey.interested')

    num_samples = len(instance_dict)
    num_trains = int(np.floor(num_samples * 0.8))
    num_tests = num_samples - num_trains

    all_list = np.random.permutation(xrange(num_samples))
    train_list = all_list[: num_trains]
    dev_list = all_list[num_trains : num_samples]

    num_epoch = 10
    dev_j = 0
    while True:
        for i in xrange(num_trains):
            ins = instance_dict[train_list[i]]
            inp = np.asarray(ins.featvec, dtype = config.floatX)
            spk_id = ins.speaker_id
            lbl = np.asarray(np.zeros((805,)), dtype = config.floatX)
            lbl[spk_id] = 1.0

            nn = NetBuilder(inp, lbl)
            nn.trainer(num_epoch)
            if i % 50 == 0:
                tins = instance_dict[dev_list[dev_j]]
                tinp = np.asarray(tins.featvec, dtype = config.floatX)
                tspk_id = tins.speaker_id
                tlbl = np.asarray(np.zeros((805,)), dtype = config.floatX)
                tlbl[tspk_id] = 1.0
                nn.tester(tinp, tlbl)
                dev_j += 1
