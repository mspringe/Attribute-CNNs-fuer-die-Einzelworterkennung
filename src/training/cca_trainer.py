"""
This module provides method to train a single RCCA estimator.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
import argparse
import random
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.join(FILE_DIR, '..', '..', ''))
sys.path.append(SRC_DIR)
sys.path.append(FILE_DIR)
from src.io import dataloader
from src.estimation import cca, cosine
from src.nn.phocnet import PHOCNet
from src.util import phoc_util, augmentation_util
from src.training.phocnet_trainer import new_logger


def parser():
    """
    Since we run training for the CCA and inference for the neural net, we need a custom/ hybrid parser

    :return: parser for CCA training
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_out', metavar='dir_out', help='directory to save the NN at')
    parser.add_argument('dset_name', metavar='dset_name', help='dataset in {iam, gw, rimes}')
    parser.add_argument('dset_csv', metavar='dset_csv', help='path to csv with meta-data')
    parser.add_argument('dset_src', metavar='dset_src', help='dir of the dataset')
    parser.add_argument('net_path', metavar='net_path', help='path to model')
    parser.add_argument('--n_code_lvl', metavar='--n_code_lvl', default='0', help='level to extract neural codes from')
    parser.add_argument('--model_name', metavar='--model_name', default='cca', help='name of your cca estimator')
    parser.add_argument('--gpu_idx', metavar='--gpu_idx', default='none', help='index of gpu_idx tu run on')
    parser.add_argument('--k_fold', metavar='--k_fold', default='1', help='number of fold to train on (gw only)')
    parser.add_argument('--augment', metavar='--augment', default='false', help='augment dataset for evaluation?')
    parser.add_argument('--alphabet', metavar='--alphabet', default='ldp', help='alphabet for PHOC')
    parser.add_argument('--scale_h', metavar='--scale_h', default='none', help='scaled height of input')
    parser.add_argument('--scale_w', metavar='--scale_w', default='none', help='scaled width of input')
    parser.add_argument('--PHOCNet_type', default='normal', help='phocnet model type {stn, normal}')
    parser.add_argument('--phoc_lvls', default=str(phoc_util.DEFAULT_PHOC_LEVELS), help='levels of the PHOC')
    return parser


def equal_split(dset : dataloader.DSetPhoc, max_sample_size):
    """
    splits the dataset into a subset of max_sample_size with evenly, but not equal distributed word classes

    :param dset: DSetPhoc object
    :param max_sample_size: maximum size of subset
    :return: subset with evenly distributed word classes
    """
    # sanity
    max_sample_size = int(max_sample_size)
    if max_sample_size >= len(dset):
        return dset
    # mapping words to idcs of occurrence
    words = list(dset.words)
    word_to_idcs = {w : [] for w in words}
    word_to_idcs_not_taken = {w : [] for w in words}
    for i, data in enumerate(dset):
        w = data['transcript']
        word_to_idcs[w].append(i)
        word_to_idcs_not_taken[w].append(True)
    # selecting idcs
    w_idx = 0
    idcs = []
    for i in range(max_sample_size):
        # word with fre idcs
        w = words[w_idx]
        while not any(word_to_idcs_not_taken[w]):
            w_idx = min(w_idx+1, len(words)-1)
            w = words[w_idx]
        # idx that has not been taken yet
        idx = 0
        while not word_to_idcs_not_taken[w][idx]:
            idx += 1
        # selecting idx and marking it as taken
        idcs.append(word_to_idcs[w][idx])
        word_to_idcs_not_taken[w][idx] = False
        # selecting next word class
        w_idx = min(w_idx+1, len(words)-1)
    # subset
    return dset.sub_set(idcs)


def gather_NC_PHOC_pairs(net, dset, logger, device, n_code_lvl=0, scale=[None, None]):
    """
    gathering neural codes and PHOC of a dataset

    :param net: a PHOCNet
    :param dset: dataset to estimate neural codes for
    :param logger: logger of information
    :param device: gpu device to place the PHOCNet on
    :param n_code_lvl: level to extract neural codes from (see :func:`src.nn.phocnet.PHOCNet.neural_codes`)
    :return: neural codes and true PHOC in that order
    """
    logger.info('processing {} images'.format(len(dset)))
    # data loader initialization
    d_loader = DataLoader(dset)
    # moving net to device, if specified
    if device is not None:
        net = net.to(device)
    net.eval()
    # gather data to perform canonical correlation analysis on
    if logger is not None:
        logger.info('gathering neural codes and respective PHOC')
    neural_codes = []
    phocs = []
    for data in d_loader:
        # gather image and respective PHOC
        img = data['img']
        img = augmentation_util.scale(img, *scale)
        phoc = data['phoc']
        # batch with size 1 for the image
        batch = torch.tensor([img.numpy()], dtype=torch.float32)
        # moving to gpu
        if device is not None:
            batch = batch.to(device)
        # extract neural codes
        n_code = net.neural_codes(batch, pos=n_code_lvl)
        # numpy conversion of neural code and PHOC
        n_code = n_code.cpu().detach().numpy()[0].astype(float)
        phoc = phoc.numpy()[0].astype(float)
        # storing vectors
        phocs.append(phoc)
        neural_codes.append(n_code)
    logger.info('done gathering')
    # freeing the gpu memory
    batch.cpu()
    net.cpu()
    # numpy conversion of samples
    neural_codes = np.array(neural_codes)
    phocs = np.array(phocs)
    return neural_codes, phocs


def run_cca_training(dset : dataloader.DSetPhoc, net : PHOCNet, words : list, device=None, n_code_lvl=0, augment=False,
                     logger=None, reg=10e3, dim=128, alphabet=phoc_util.DEFAULT_ALPHABET):
    """
    Training a cca on a dataset, given a PHOCNet.

    :param dset: dataset to train the RCCA on
    :param net: a PHOCNet
    :param words: lexicon of words
    :param device: gpu device to place the PHOCNet on
    :param n_code_lvl: level to extract neural codes from (see :func:`src.nn.phocnet.PHOCNet.neural_codes`)
    :param augment: whether to augment images
    :param logger: logger of information
    :param reg: regularization hyper parameter for RCCA
    :param dim: dimension hyper parameter for RCCA
    :param alphabet: alphabet used
    :return: trained RCCA
    """
    # augmentation
    if augment:
        dset.augment_imgs = True
    # gather NC and PHOC
    neural_codes, phocs = gather_NC_PHOC_pairs(net=net, dset=dset, logger=logger, device=device, n_code_lvl=n_code_lvl)
    # train CCA, based on NC and PHOC
    est_cca = train_cca(neural_codes=neural_codes, phocs=phocs, dim=dim, reg=reg, logger=logger, words=words,
                        alphabet=alphabet)
    return est_cca

def train_cca(neural_codes, phocs, dim, reg, logger, words, alphabet=phoc_util.DEFAULT_ALPHABET):
    """
    Training a RCCA, given neural codes and respective PHOC

    :param neural_codes: neural codes
    :param phocs: respective PHOC
    :param dim: dimension hyper parameter for RCCA
    :param reg: regularization hyper parameter for RCCA
    :param logger: logger of information
    :param words: lexicon of words
    :param alphabet: alphabet used
    :return: trained RCCA
    """
    # initialize estimator
    est_cca = cca.RCCAEstimator(words=words, n_dim=dim, reg=reg, alphabet=alphabet)
    # CCA
    if logger is not None:
        logger.info('Training CCA, sample size: {}'.format(len(neural_codes)))
    est_cca.fit(neural_codes, phocs)
    if logger is not None:
        logger.info('Finished training')
    return est_cca
