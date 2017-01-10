#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import sys
import os
import re

import numpy as np
import theano as T
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Activation, \
    MaxPooling2D, ZeroPadding2D, Dropout, Flatten, LSTM, Input, \
    Reshape
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

speaker_dict = dict()
instance_dict = dict()

class Instance:
    def __init__(self, file_id, speaker_id, featvec):
        self.file_id = file_id
        self.speaker_id = speaker_id
        self.featvec = featvec

def read_instance(data_path, filename):
    """
    Read data as instances.

    :data_path: (string) path containing data to be processed
    :filename: (string) file containing names of interested files
    """

    filelist = [l for l in open(filename)] # all files in filename
    interest_filelist = [''.join([ l.split()[0].split('.')[0], '.txt' ])
                         for l in filelist] # filenames of interest
    num_files = len(filelist)
    for i in xrange(num_files):
        name = filelist[i]
        speaker_id = name.split()[1]
        if speaker_id in speaker_dict:
            speaker_id = speaker_dict[speaker_id]
        else:
            speaker_dict[speaker_id] = len(speaker_dict)
            speaker_id = speaker_dict[speaker_id]

        file = interest_filelist[i]
        featvec = [ line.strip().split(',')
                   for line in open(os.path.join(data_path,file)) ]
        featvec = np.asarray(featvec, dtype=float)
        instance_dict[i] = Instance(file, speaker_id, featvec)
    write_dict(speaker_dict, ''.join([filename, '_consq_speaker_dict.txt']))
    print('processed ', filename, ', total ', num_files, ' files')
    print('total ', len(speaker_dict), ' speakers')

def write_dict(speaker_dict, filename):
    out = ''
    for k in speaker_dict:
        out += str(k) + ',' + str(speaker_dict[k]) + '\n' # k, v
    with open(filename, 'w') as f:
        f.write(out)

class NetBuilder:
    def __init__(self, inp, lbl):
        nrow, ncol = inp.shape
        nout = lbl.shape
        inp = inp.reshape(1, 1, nrow, ncol)
        self.input = inp
        self.output = lbl
        self.model = Sequential()
        self.model.add(Convolution2D(
            10, nrow, 1,
            input_shape = (1, nrow, ncol),
            subsample = (1,1)
        )
                       )
        self.model.add(Activation('relu'))
        # self.model.add(Reshape((10, -1)))
        # self.model.add(LSTM(32))
        self.model.add(Flatten())
        self.model.add(Dense(nout))
        self.model.add(Activation('softmax'))

    def trainer(self, nepoch):
        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer = 'sgd',
            metrics = ['accuracy']
        )
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

    def tester(self):
        pass

    def saver(self):
        pass


if __name__ == '__main__':
    read_instance('./feat_constq/ey/', './ey.interested')
    num_samples = len(instance_dict)
    num_trains = int(np.floor(num_samples * 0.8))
    num_tests = num_samples - num_trains
    train_list = np.random.permutation(xrange(num_trains))
    # dev_list =
    while True:
        for i in xrange(num_trains):
            ins = instance_dict[train_list[i]]
            inp = np.asarray(ins.featvec, dtype = T.config.floatX)
            lbl = np.array([ins.speaker_id])
            nn = NetBuilder(inp, lbl)
            nn.trainer(num_epoch)
            # nn.tester()
