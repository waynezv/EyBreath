#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function
import os
import sys

import numpy as np
from dynet import *

import constants
import data_prepare

class NetworkBuilder(object):
    pass

if __name__ == '__main__':
    data_prepare.read_instance(constants.DATA_PATH1, constants.EY_FILE_LIST)
