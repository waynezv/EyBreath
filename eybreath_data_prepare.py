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

instance_dict = dict() # store 'speaker_id (string) : instances (list of
                    # instance objects)' pairs
instance_collection = [] # collection of instances

# TODO:
def normalization():
    """
    Normalize input features.
    """
    pass

def elastic_transform(image, alpha, sigma, random_state=None):
    """
    Data augmentation by elastic transform.
    """
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
    One instance.
    """
    def __init__(self, file_id, speaker_id, featvec):
        self.file_id = file_id
        self.speaker_id = speaker_id
        self.featvec = featvec

def read_instance(data_path, filelist_name,
        num_tests=None,
        use_dct=False,
        use_unknown=False, unknown_class=None,
        distort=False, sigma=None, alpha=None, ds_limit=5):
    """
    Read data as instances.

    :data_path: (string) path containing data to be processed
    :filelist_name: (string) file containing list of filenames
    :num_tests: (int) number of test samples, for sampling test data only
    :use_dct: (bool)
    :use_unknown: (bool)
    :unknown_class: (int)
    :distort: (bool)
    :sigma, alpha: (float) elastic transform parameters
    :ds_limit: (int) number of distorsion
    """

    # Read all instances
    filelist = [l for l in open(filelist_name)] # read in filelist_name
    interest_filelist = [l.split()[0] for l in filelist] # files of interest

    if num_tests is not None:
        num_files = num_tests
        read_list = np.random.permutation(len(interest_filelist))[:num_files]

    else:
        num_files = len(interest_filelist)
        read_list = range(num_files)

    for i in read_list:
        line = filelist[i]
        sspk_id = line.split()[1]
        speaker_id = int(line.split()[2])

        file = interest_filelist[i]
        featvec = np.asarray([
            line.strip().split(',')
            for line in open(os.path.join(data_path, file))
        ], dtype = 'float')

        if sspk_id not in instance_dict:
            instance_dict[sspk_id] = [] # init speaker instance list

        if distort:
            rds = np.random.RandomState(12345)
            for _ in xrange(ds_limit):
                featvec = elastic_transform(featvec, alpha, sigma, random_state=rds)
                instance_dict[sspk_id].append( Instance(file, speaker_id, featvec) )

        elif dct: # deprecated
            coeff = dct(dct(featvec, axis=0, norm='ortho'), axis=1, norm='ortho')
            coeff[0:15, 0:15] = 0.
            featvec = idct(idct(coeff, axis=1), axis=0)
            instance_dict[sspk_id].append( Instance(file, speaker_id, featvec) )

        else:
            instance_dict[sspk_id].append( Instance(file, speaker_id, featvec) )

    for k in instance_dict:
        instance_collection.append(instance_dict[k])

    print('processed ', filelist_name, ', total ', num_files, ' files')

    # TODO: unknown

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

def load_data(dpath, filelist_name, shuffle = False,
        num_tests=None,
        use_unknown=False, unknown_class=None,
        use_dct=False,
        distort=False, sigma=None, alpha=None, ds_limit=5):
    """
    Load data.

    <- train_set, val_set, test_set: [X (list of 4d tensors), y (list of integers)]
    or
    <- test_set only
    """
    # Read instances from constant Q features
    if num_tests is None: # use all data
        read_instance(dpath, filelist_name,
                num_tests=num_tests,
                use_unknown=use_unknown, unknown_class=unknown_class,
                use_dct=use_dct,
                distort=distort, sigma=sigma, alpha=alpha, ds_limit=ds_limit)

        # Split dataset
        train_set = []
        val_set = []
        test_set = []
        split_ratio = (0.7, 0.2, 0.1)

        for i in xrange(len(instance_dict)):
            ins_list = instance_collection[i]

            num_ins = len(ins_list)
            num_trns = int(np.floor(num_ins * split_ratio[0]))
            num_vals = int(np.floor(num_ins * split_ratio[1]))
            num_tes = int(num_ins - num_trns - num_vals)

            smpl_list = np.arange(num_ins)

            if shuffle: # within speaker shuffle
                np.random.shuffle(smpl_list)

            trn_list = smpl_list[: num_trns]
            val_list = smpl_list[num_trns : num_trns + num_vals]
            tes_list = smpl_list[num_trns + num_vals : num_ins]

            xy_trn = prepare_data(ins_list, trn_list)
            xy_val = prepare_data(ins_list, val_list)
            xy_tes = prepare_data(ins_list, tes_list)

            for x, y in xy_trn:
                train_set.append((x, y))
            for x, y in xy_val:
                val_set.append((x, y))
            for x, y in xy_tes:
                test_set.append((x, y))

        if shuffle: # among speaker / global shuffle
            trf = np.random.permutation(len(train_set))
            vlf = np.random.permutation(len(val_set))
            tef = np.random.permutation(len(test_set))

            train_set = [train_set[s] for s in trf]
            val_set = [val_set[s] for s in vlf]
            test_set = [test_set[s] for s in tef]

        return train_set, val_set, test_set

    else: # use only test data
        read_instance(dpath, filelist_name,
                num_tests=num_tests, use_unknown=use_unknown, unknown_class=unknown_class,
                use_dct=use_dct,
                distort=distort, sigma=sigma, alpha=alpha, ds_limit=ds_limit)

        test_set = []
        print('num test speakers: ', len(instance_dict))

        for i in xrange(len(instance_dict)):
            ins_list = instance_collection[i]

            smpl_list = np.arange(len(ins_list))
            if shuffle:
                np.random.shuffle(smpl_list)

            xy_ = prepare_data(ins_list, smpl_list)

            for x, y in xy_:
                test_set.append((x, y)) # caveat: if use generator may cause error:
                                    # test_set.append((x,y) for x,y in xy_)
        if shuffle:
            tef = np.random.permutation(len(test_set))
            test_set = [test_set[s] for s in tef]

        return test_set


if __name__ == '__main__':
    dpath = '../feat_constq'

    # train_set, val_set, test_set = load_data(
        # os.path.join(dpath ,'ey'), './ey_selected_100', shuffle = True,
        # num_tests=None,
        # use_unknown=False, unknown_class=None,
        # use_dct=False,
        # distort=False, sigma=None, alpha=None, ds_limit=5)

    test_set = load_data(
        os.path.join(dpath ,'ey'), './ey_selected_100', shuffle = True,
        num_tests=1000,
        use_unknown=False, unknown_class=None,
        use_dct=False,
        distort=False, sigma=None, alpha=None, ds_limit=5)
