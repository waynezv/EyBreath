"""
Data preparation for speaker identification with Ey Breath.
"""
from __future__ import print_function
import os
import sys

import numpy as np
from scipy.fftpack import dct, idct
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from theano import config

speaker_dict = dict() # store 'speaker_id (string) : id (integer)' pairs
instance_dict = dict() # store 'speaker_id (string) : instances (list of
    # instance objects)' pairs
count_dict = dict() # store 'speaker_id (string) : counter (integer)' pairs
instance_collection = [] # collection of instances

# TODO:
def normalization():
    """
    Normalize input features.
    """
    pass

def elastic_transform(image, alpha, sigma, random_state=None):
    assert len(image.shape)==2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    return map_coordinates(image, indices, order=3, mode="nearest").reshape(shape)

class Instance:
    """
    One training instance.
    """
    def __init__(self, file_id, speaker_id, featvec):
        self.file_id = file_id
        self.speaker_id = speaker_id
        self.featvec = featvec

def read_instance(data_path, filename, spk_smpl_thrd = 100,
        use_dct=False,
        use_unknown=False, unknown_class=None,
        distort=False, sigma=None, alpha=None, ds_limit=5,
        wrt_dict = False):
    """
    Read data as instances.

    :data_path: (string) path containing data to be processed
    :filename: (string) file containing names of interested files
    :spk_smpl_thrd: (integer) only speaker that has samples exceeds
    this number is counted in
    :wrt_dict: (bool) write speaker_dict to file or not
    """

    filelist = [l for l in open(filename)] # all files in filename
    interest_filelist = [''.join([ l.split()[0].split('.')[0], '.txt' ])
                         for l in filelist] # filenames of interest
    num_files = len(filelist)
    for i in xrange(num_files):
        name = filelist[i]
        sspk_id = name.split()[1]
        if sspk_id in speaker_dict: # if speaker in dict
            count_dict[sspk_id] += 1 # add counter
            speaker_id = speaker_dict[sspk_id] # get mapped id

        else: # if speaker shows up first time
            count_dict[sspk_id] = 1 # set counter
            speaker_dict[sspk_id] = len(speaker_dict)
            speaker_id = speaker_dict[sspk_id]
            instance_dict[sspk_id] = [] # init speaker instance list

        file = interest_filelist[i]
        featvec = np.asarray([
            line.strip().split(',')
            for line in open(os.path.join(data_path, file))
        ], dtype = 'float')

        if distort:
            rds = np.random.RandomState(1234)
            for _ in xrange(ds_limit):
                featvec = elastic_transform(featvec, alpha, sigma, random_state=rds)
                instance_dict[sspk_id].append( Instance(file, speaker_id, featvec) )

        if dct:
            coeff = dct(dct(featvec, axis=0, norm='ortho'), axis=1, norm='ortho')
            coeff[0:15, 0:15] = 0.
            featvec = idct(idct(coeff, axis=1), axis=0)
            #featvec = (featvec - np.amin(featvec)) / \
            #        (np.amax(featvec) - np.amin(featvec))
            instance_dict[sspk_id].append( Instance(file, speaker_id, featvec) )

        else:
            instance_dict[sspk_id].append( Instance(file, speaker_id, featvec) )

    enrolled_ins_count = 0 # counter for enrolled instances
    sid = 0
    for k in count_dict: # for each speaker
        if count_dict[k] >= spk_smpl_thrd: # if exceeds threshold
            ins_list = instance_dict[k]
            enrolled_ins_count += len(ins_list)
            for r in ins_list:
                r.speaker_id = int(sid) # reprog speaker id
            instance_collection.append(ins_list) # add instances to collection
            sid += 1
        elif use_unknown: # put in 'unknown' class
            ins_list = instance_dict[k]
            for r in ins_list:
                r.speaker_id = unknown_class # TODO: out of set classes
            instance_collection.append(ins_list)
    num_enrolled_spk = len(instance_collection) # number of enrolled speakers

    if wrt_dict:
        write_dict(speaker_dict, ''.join([filename, '_consq_speaker_dict.txt']))
        print('Saved speaker_dict to file: ', filename + '_consq_speaker_dict.txt')

    print('processed ', filename, ', total ', num_files, ' files')
    print('total ', len(speaker_dict), ' speakers')
    print('totally enrolled ', enrolled_ins_count, ' instances, ',
          num_enrolled_spk, ' speakers.')

    return num_enrolled_spk, enrolled_ins_count

def write_dict(speaker_dict, filename):
    out = ''
    for k in speaker_dict:
        out += str(k) + ',' + str(speaker_dict[k]) + '\n' # k, v
    with open(filename, 'w') as f:
        f.write(out)

def prepare_data(ins_list, idx_list):
    """
    Format data to network required form.
    :ins_list: list of instances
    :idx_list: list of indices
    <- [X (list of 4d tensors), y (list of integers)]
    """

    X = []
    y = []
    for idx in idx_list:
        ins = ins_list[idx]
        spk_id = ins.speaker_id
        y.append(spk_id)
        ins_vec = ins.featvec
        nrow, ncol = ins_vec.shape
        inp = np.asarray(ins_vec, dtype = config.floatX).reshape(
            (1, 1, nrow, ncol))
        X.append(inp)

    return zip(X, y)

def load_data(dpath, filename, shuffle = False, spk_smpl_thrd = 100,
        use_unknown=False, unknown_class=None,
        use_dct=False,
        distort=False, sigma=None, alpha=None, ds_limit=5):
    """
    Load data.
    <- train_set, val_set, test_set: [X (list of 4d tensors), y (list of integers)]
    """
    # Read instances from constant Q features
    num_enrolled_spk, enrolled_ins_count = read_instance(dpath, filename,
            spk_smpl_thrd=spk_smpl_thrd, use_unknown=use_unknown, unknown_class=unknown_class,
            use_dct=use_dct,
            distort=distort, sigma=sigma, alpha=alpha, ds_limit=ds_limit)

    # Split dataset
    train_set = []
    val_set = []
    test_set = []
    split_ratio = (0.7, 0.2, 0.1)
    for i in xrange(num_enrolled_spk):
        ins_list = instance_collection[i]
        num_ins = len(ins_list)
        num_trns = int(np.floor(num_ins * split_ratio[0]))
        num_vals = int(np.floor(num_ins * split_ratio[1]))
        num_devs = int(num_ins - num_trns - num_vals)

        smpl_list = np.arange(num_ins)

        if shuffle:
            np.random.shuffle(smpl_list)

        trn_list = smpl_list[: num_trns]
        val_list = smpl_list[num_trns : num_trns + num_vals]
        dev_list = smpl_list[num_trns + num_vals : num_ins]

        xy_trn = prepare_data(ins_list, trn_list)
        xy_val = prepare_data(ins_list, val_list)
        xy_dev = prepare_data(ins_list, dev_list)

        for x, y in xy_trn:
            train_set.append((x, y))
        for x, y in xy_val:
            val_set.append((x, y))
        for x, y in xy_dev:
            test_set.append((x, y))

    if shuffle:
        tlf = np.random.permutation(len(train_set))
        vlf = np.random.permutation(len(val_set))
        dlf = np.random.permutation(len(test_set))
        train_set = [train_set[s] for s in tlf]
        val_set = [val_set[s] for s in vlf]
        test_set = [test_set[s] for s in dlf]

    return train_set, val_set, test_set


if __name__ == '__main__':
    #dpath = './feat_constq'
    dpath = '/mingback/zhaowenbo/EyBreath/feat_constq'
    train_set, val_set, test_set = load_data(os.path.join(dpath ,'ey'), './ey.interested', shuffle = True, spk_smpl_thrd = 100)
