"""
This module provides a method, that outputs all unique characters of a dataset.

It can be used as a script, to print mentioned characters of a dataset:

::

    python3 alphabet_chars.py gw path/to/annotations path/to/images


This can be usefull, when adapting alphabets of new data sets to this framework.


.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
import sys
import os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.join(FILE_DIR, '..', '..', ''))
sys.path.append(SRC_DIR)
from argparse import ArgumentParser
import src.io.dataloader as dl


def parser():
    parser = ArgumentParser()
    parser.add_argument('dataset', help='name of dataset')
    parser.add_argument('p_csv', help='path to groundtruth')
    parser.add_argument('p_imgs', help='path to images')
    return parser


def dset_chars(dset: dl.GWDataSet):
    """
    Returns unique characters of the data set

    :param dset: data set
    :return: the data sets unique characters
    """
    words = list(dset.words)
    all_chars = ''.join(words)
    set_chars = ''.join(sorted(set(all_chars)))
    return set_chars


if __name__ == '__main__':
    parser = parser()
    args = vars(parser.parse_args())
    str_dset = args['dataset']
    p_csv = args['p_csv']
    p_imgs = args['p_imgs']
    if str_dset == 'gw':
        dset = dl.GWDataSet(csvs_path=p_csv, imgs_path=p_imgs)
    elif str_dset == 'iam':
        dset = dl.IAMDataset(csvs_path=p_csv, imgs_path=p_imgs)
    elif str_dset == 'rimes':
        dset = dl.RimesDataSet(csvs_path=p_csv, imgs_path=p_imgs)
    print(dset_chars(dset))
