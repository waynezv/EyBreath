from __future__ import print_function
import os
import sys

import numpy as np
import constants

instance_dict = dict()
speaker_dict = dict()
class Instance(object):
    def __init__(self, file_id, speaker_id, featvec, num_frames):
        self.file_id = file_id
        self.speaker_id = speaker_id
        self.featvec = featvec
        self.num_frames = num_frames

def read_instance(data_path, filename, pad_size, instance_save_dir):
    filelist = [l for l in open(filename)] # all files in filename
    interest_filelist = [''.join([l.split()[0].split('.')[0],
                                  '.80-7200_40filts.lspec.ascii'])
                         for l in filelist] # filenames of interest
    num_files = len(filelist)
    max_frames = 0
    for i in xrange(num_files):
        name = filelist[i]
        file = interest_filelist[i]
        speaker_id = ''.join(name.split()[1].split('_')[:2])
        if speaker_id in speaker_dict:
            speaker_id = speaker_dict[speaker_id]
        else:
            speaker_dict[speaker_id] = len(speaker_dict)
            speaker_id = speaker_dict[speaker_id]
        with open(os.path.join(data_path,file)) as f:
            header = f.readline()
            subheader = f.readline()
            num_frames = int(subheader.split()[1])
            if num_frames > max_frames:
                max_frames = num_frames
            featvec = []
            for rstl in f:
                featvec.append(rstl.strip().split()[1:41]) # strip first number
            featvec = pad_vec(pad_size, featvec) # padding
            featvec = np.asarray(featvec, dtype=float)
            instance_dict[speaker_id] = Instance(file, speaker_id, featvec, num_frames)
    # write_dict(speaker_dict, ''.join([filename, '_speaker_dict.txt']))
    # write_instance(instance_dict, instance_save_dir)
    print('processed ', filename, ', total ', num_files, ' files')
    print('total ', len(speaker_dict), ' speakers')
    print('max frame number ', max_frames)

def pad_vec(max_frames, vec):
    v = np.zeros((max_frames, 40))
    r, c = np.array(vec).shape
    if r < max_frames:
        v[:r] = vec
        return v
    else:
        return vec[:max_frames]

def write_instance(instance_dict, path):
    for k in instance_dict:
        data = ''
        data += str(k) + '\n' + str(instance_dict[k].num_frames) + '\n'
        for l in instance_dict[k].featvec:
            data += str(l).strip() + '\n'
        with open(''.join([path, str(k), '.txt']), 'w') as f:
            f.write(data)

def write_dict(speaker_dict, filename):
    out = ''
    for k in speaker_dict:
        out += str(k) + ',' + str(speaker_dict[k]) + '\n' # k, v
    with open(filename, 'w') as f:
        f.write(out)



if __name__ == '__main__':
    # read_instance(constants.DATA_PATH1, constants.EY_FILE_LIST, constants.MAX_FRAMES1, 'featvec_ey/')
    # read_instance(constants.DATA_PATH2, constants.BREATH_FILE_LIST, constants.MAX_FRAMES2, 'featvec_breath/')
