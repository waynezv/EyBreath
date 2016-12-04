from __future__ import print_function
import os
import sys

import numpy as np
from dynet import *

DATA_PATH = './logspec'
EY_FILE_LIST = './ey.interested'
BREATH_FILE_LIST = './breath.interested'

instance_dict = dict()
class Instance(object):
    def __init__(self, file_id, speaker_id, featvec, num_frames):
        self.file_id = file_id
        self.speaker_id = speaker_id
        self.featvec = featvec
        self.num_frames = num_frames

def read_instance():
    lines = [l for l in open(EY_FILE_LIST)]
    interest_filelist = [''.join(l.split()[0].split('.')[0],'.80-7200_40filts.lspec.ascii')
                         for l in lines]
    for file in interest_filelist:
        speaker_id = ''.join(file.split()[1].split('_')[:2])
        with open(''.join(DATA_PATH,file)) as f:
            header = f.readline()
            subheader = f.readline()
            num_frames = int(subheader.split()[1])
            featvec = f.readlines()
            instance_dict[speaker_id] = Instance(file, speaker_id, featvec, num_frames)


if __name__ == '__main__':
    read_instance()
