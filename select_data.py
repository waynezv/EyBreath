#!/usr/bin/env python
# encoding: utf-8

"""
select data for speaker identification with Ey Breath.
"""
from __future__ import print_function
import os
import sys
import numpy as np

speaker_dict = dict() # store 'speaker_id (string) : id (integer)' pairs
count_dict = dict() # store 'speaker_id (string) : counter (integer)' pairs
instance_dict = dict() # store 'speaker_id (string) : instances (list of
                    # instance objects)' pairs

class Instance:
    """
    One instance.
    """
    def __init__(self, file_id, speaker_id):
        self.file_id = file_id
        self.speaker_id = speaker_id

def write_dict(speaker_dict, filename):
    """
    Save [speaker name(str) : speaker id(int)] pairs.
    """
    out = ''
    for k in speaker_dict:
        out += str(k) + ',' + str(speaker_dict[k]) + '\n' # k, v
    with open(filename, 'w') as f:
        f.write(out)

def write_data_list(filename, data_list):
    """
    Save selected data_list.
    """
    out = ''

    for fn in data_list:
        out += fn + '\n'

    with open(filename, 'w') as f:
        f.write(out)

def select_data(data_path, filename, spk_smpl_thrd = 100, write_dict=False):
    """
    Select speakers and instances that exceeds threshold.
    """

    # Read all instances
    filelist = [l for l in open(filename)] # all files in filename
    interest_filelist = [''.join([ l.split()[0].split('.')[0], '.txt' ])
                         for l in filelist] # files of interest

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

        instance_dict[sspk_id].append( Instance(file, speaker_id) )

    # Select instances
    num_enrolled_ins = 0 # number of enrolled instances
    num_enrolled_spk = 0 # number of enrolled speakers

    data_list = [] # list of enrolled files
    unknown_list = [] # list of out-of-set files

    sid = 0 # speaker id
    for k in count_dict: # for each speaker
        if count_dict[k] >= spk_smpl_thrd: # if exceeds threshold
            ins_list = instance_dict[k]
            data_list.extend(''.join([ins.file_id, ' ', k, ' ', str(sid)]) for ins in ins_list)

            num_enrolled_spk += 1
            sid += 1

        else: # put in unknown
            ins_list = instance_dict[k]
            unknown_list.extend(''.join([ins.file_id, ' ', k]) for ins in ins_list)

    print('processed ', filename, ', total ', num_files, ' files')
    print('total ', len(speaker_dict), ' speakers')
    print('totally enrolled ', len(data_list), ' instances, ',
          num_enrolled_spk, ' speakers.')
    print('out-of-set ', len(unknown_list), ' instances, ',
          len(speaker_dict)-num_enrolled_spk, ' speakers.')

    # Save list
    sel_name = ''.join(['ey_selected_', str(spk_smpl_thrd)])
    unk_name = ''.join(['ey_unknown_', str(spk_smpl_thrd)])

    write_data_list(sel_name, data_list)
    write_data_list(unk_name, unknown_list)

    print('Saved selected filelist to ', sel_name)
    print('Saved out-of-set filelist to ', unk_name)

    if wrt_dict:
        dict_name = ''.join([filename, '_consq_speaker_dict.txt'])
        write_dict(speaker_dict, dict_name)
        print('Saved speaker_dict to file: ', dict_name)


if __name__ == '__main__':
    dpath = '../feat_constq'
    select_data(os.path.join(dpath ,'ey'), './ey.interested', spk_smpl_thrd = 100)
