"""
This module contains methods for sanity of file-names and directories.
These methods can be useful, when you're writing to disk.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
import os
import numpy as np


def safe_dir_path(dir_path):
    """makes sure directory exists"""
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


def unique_file_name(dir, fn, suffix):
    """adds enumeration to filename fn, iff the filename has been taken already"""
    file_name = os.path.join(dir, fn + suffix)
    i = 1
    while os.path.isfile(file_name):
        file_name = os.path.join(dir, fn + '({})'.format(i) + suffix)
        i += 1
    return file_name


def np_arr(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x
