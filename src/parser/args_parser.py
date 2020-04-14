"""
This module provides args-parser for training and inference.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
import argparse
import os
import sys
# own code
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.join(FILE_DIR, '..', '..', ''))
sys.path.append(SRC_DIR)
from src.util import phoc_util


def parser_training():
    """
    This method provides an args-parser for training arguments.

    :return: an args parser for the following arguments

        Positional:

        =============== ======================================================
        arg             semantic
        =============== ======================================================
        dir_out         the directory to safe the net to
        dset_name       the dataset to train on
        dset_csv        path to the csv(/dir) with metadata
        dset_src        dir of the dataset(-images)
        =============== ======================================================

        Optional:

        =============== =================================================================================== ============
        arg             semantic                                                                            default
        =============== =================================================================================== ============
        --gpu_idx       name of gpu_idx-device to run on (cuda:X)                                           none
        --max_iter      maximum number of batches processed during training                                 100,000
        --loss          loss function used in training                                                      cosine
        --augment       indicates augmentation of dataset                                                   eq
        --model_name    name/ id for the model                                                              my_model
        --optimizer     choose between adam and sgd optimizer                                               adam
        --PHOCNet_type  choose between my (classic) implementation and an extra STN layer                   normal
        --k_fold        number of folding index for Almazans cross validation on the GW set                 1
        --stop_words    flag indicating whether to use stop words for IAM-DB                                true
        --punctuation   flag indicating whether to use punctuation for IAM-DB                               true
        --lr            set the initial learning rate manually                                              0.0001
        --save_interval interval of iterations/ frequency to save a statedict during training               10,000
        --alphabet      alphabet properties to be utilized (see :func:`src.util.phoc_util.rep_to_alphabet`) ldp
        --phoc_lvls     number of levels used for the PHOC                                                  3
        --s_batch       batch size                                                                          10
        --shuffle       shuffling of train data                                                             true
        --scale_w       scale of image, no scaling along axis if none                                       none
        --scale_h       scale of image, no scaling along axis if none                                       none
        --pretrained    path to a pretrained model                                                          none
        =============== =================================================================================== ============
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_out', help='directory to save the NN at')
    parser.add_argument('dset_name',  help='dataset in {iam, gw, rimes, hws}')
    parser.add_argument('dset_csv', help='path to csv with meta-data')
    parser.add_argument('dset_src', help='dir of the dataset')
    parser.add_argument('--gpu_idx', default='none', help='index of gpu_idx tu run on')
    parser.add_argument('--max_iter', default='100000', help='number of maximum iterations/ steps')
    parser.add_argument('--loss', default='cosine', help='loss function to be used in training')
    parser.add_argument('--augment', default='eq', help='indication of augmentation for training')
    parser.add_argument('--model_name', default='my_model', help='name of the trained model')
    parser.add_argument('--optimizer', default='adam', help='optimzer to be used (adam, sgd), default: adam')
    parser.add_argument('--PHOCNet_type', default='normal', help='phocnet model type {stn, normal}')
    parser.add_argument('--k_fold', default='1', help='fold number (e.g. relevant for gw)')
    parser.add_argument('--stop_words', default='true', help='indicates whether to include stop words in training data')
    parser.add_argument('--punctuation', default='true',
                        help='indicates whether to include punctuation in training data')
    parser.add_argument('--lr', default='0.0001', help='learning rate of optimizer')
    parser.add_argument('--save_interval', default='10000', help='interval to save statedict of model')
    parser.add_argument('--alphabet', default='ldp',
                        help='alphabet to be used l: lowercase, u: uppercase, d: digits, p: punctuation')
    parser.add_argument('--phoc_lvls', default=str(phoc_util.DEFAULT_PHOC_LEVELS), help='levels of the PHOC')
    parser.add_argument('--s_batch', default='10', help='batch size')
    parser.add_argument('--shuffle', default='true', help='shuffling of train-data')
    parser.add_argument('--scale_w', default='none', help='scaling of images')
    parser.add_argument('--scale_h', default='none', help='scaling of images')
    parser.add_argument('--pretrained', default='none', help='path to a pretrained model')
    return parser


def parser_inference():
    """
    This method provides an args-parser for inference arguments.

    :return: an args parser for the following arguments

        Positional:

        =============== ======================================================
        arg             semantic
        =============== ======================================================
        net_path        path to the PHOCNets state-dict
        dir_out         the directory to safe the net to
        dset_name       the dataset to train on
        dset_csv        path to the csv(/dir) with metadata
        dset_src        dir of the dataset(-images)
        =============== ======================================================

        Optional:

        =============== =================================================================================== ============
        arg             semantic                                                                            default
        =============== =================================================================================== ============
        --estimator     kind of estimator/ path to pickled estimator                                        cosine
        --gpu_idx       name of gpu_idx-device to run on (cuda:X)                                           none
        --model_name    name/ id for the model                                                              my_model
        --PHOCNet_type  choose between my (classic) implementation and an extra STN layer                   normal
        --k_fold        number of folding index for Almazans cross validation on the GW set                 1
        --stop_words    flag indicating whether to use stop words for IAM-DB                                true
        --alphabet      alphabet properties to be utilized (see :func:`src.util.phoc_util.rep_to_alphabet`) ldp
        --phoc_lvls     number of levels used for the PHOC                                                  3
        --s_batch       batch size                                                                          10
        --scale_w       scale of image, no scaling along axis if none                                       none
        --scale_h       scale of image, no scaling along axis if none                                       none
        --n_codes_lvl   level to extract neural codes from. Ranging from -4 to 0=output                     0
        =============== =================================================================================== ============
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('net_path', help='path to the PHOCNets state-dict')
    parser.add_argument('dir_out', help='directory to save the NN at')
    parser.add_argument('dset_name', help='dataset in {iam, gw, rimes, hws}')
    parser.add_argument('dset_csv', help='path to csv with meta-data')
    parser.add_argument('dset_src', help='dir of the dataset')
    parser.add_argument('--estimator', default='cosine', help='kind of estimator/ path to pickled estimator')
    parser.add_argument('--gpu_idx', default='none', help='index of gpu_idx tu run on')
    parser.add_argument('--model_name', default='my_model', help='name of the trained model')
    parser.add_argument('--PHOCNet_type', default='normal', help='phocnet model type {stn, normal}')
    parser.add_argument('--k_fold', default='1', help='fold number (e.g. relevant for gw)')
    parser.add_argument('--stop_words', default='true', help='indicates whether to include stop words in training data')
    parser.add_argument('--alphabet', default='ldp',
                        help='alphabet to be used l: lowercase, u: uppercase, d: digits, p: punctuation')
    parser.add_argument('--phoc_lvls', default=str(phoc_util.DEFAULT_PHOC_LEVELS), help='levels of the PHOC')
    parser.add_argument('--s_batch', default='10', help='batch size')
    parser.add_argument('--scale_w', default='none', help='scaling of images')
    parser.add_argument('--scale_h', default='none', help='scaling of images')
    parser.add_argument('--n_codes_lvl', default='0',
                        help='level to extract neural codes from. Ranging from -4 to 0=output')
    return parser
