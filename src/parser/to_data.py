"""
This module provides methods, that ease the interpretation of passed arguments.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
import os
import sys
import pickle
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.join(FILE_DIR, '..', '..', ''))
sys.path.append(SRC_DIR)
from src.estimation import cca, cosine, prob, euclidean
from src.nn.phocnet import *
from src.io import dataloader
from src.util import phoc_util


def get_estimator(est_name, words, alphabet):
    """
    establishing the estimator

    :param est_name: name of estimator or path to pickled estimator
    :param words: lexicon of words
    :param alphabet: used alphabet
    :return: specified estimator
    """
    # cosine distance
    if est_name == 'cosine':
        estimator = cosine.CosineEstimator(words=words, alphabet=alphabet)
    # PRM-score
    elif est_name == 'prob':
        estimator = prob.ProbEstimator(words=words, alphabet=alphabet)
    # euclidean distance
    elif est_name == 'euclidean':
        estimator = euclidean.EuclideanEstimator(words=words, alphabet=alphabet)
    # RCCA
    else:
        if os.path.isfile(est_name):
            with open(est_name, 'rb') as f_est:
                estimator = pickle.load(f_est)
        else:
            raise ValueError('unknown estimator: {}'.format(est_name))
    return estimator


def get_dsets(dset_name, dset_csv, dset_imgs, alphabet, scale, k_fold=1, phoc_lvls=phoc_util.DEFAULT_PHOC_LEVELS):
    """
    etsablishing the dataset

    :param dset_name: name of the dataset
    :param dset_csv: path to the annotations
    :param dset_imgs: path to the images
    :param alphabet: used alphabet
    :param scale: scaling of images
    :param k_fold: number of fold (in case of the George Washington dataset)
    :param phoc_lvls: levels of PHOC
    :return: specified dataset, aswell as training- and test-split
    """
    lower_case = dataloader.Alphabet.ASCII_UPPER not in alphabet
    if dset_name == 'iam':
        data_set = dataloader.IAMDataset(csvs_path=dset_csv, imgs_path=dset_imgs, alphabet=alphabet,
                                         lower_case=lower_case, scale=scale, phoc_levels=phoc_lvls)
        train, test = data_set.train_test_official()
    elif dset_name == 'gw':
        data_set = dataloader.GWDataSet(csvs_path=dset_csv, imgs_path=dset_imgs, alphabet=alphabet,
                                        lower_case=lower_case, scale=scale, phoc_levels=phoc_lvls)
        train, test = data_set.fold(k=k_fold)
    elif dset_name == 'rimes':
        data_set = dataloader.RimesDataSet(csvs_path=dset_csv, imgs_path=dset_imgs, alphabet=alphabet,
                                           lower_case=lower_case, scale=scale, phoc_levels=phoc_lvls)
        train, test = data_set.train_test_official()
    elif dset_name == 'hws':
        data_set = dataloader.HWSynthDataSet(csvs_path=dset_csv, imgs_path=dset_imgs, alphabet=alphabet,
                                             lower_case=lower_case, scale=scale, phoc_levels=phoc_lvls)
        train, test = data_set.train_test_official()
    else:
        raise NotImplementedError('only iam, gw, rimes and hws dataset')
    return data_set, train, test


def get_PHOCNet(t_phocnet, alphabet, phoc_lvls=phoc_util.DEFAULT_PHOC_LEVELS):
    """
    establishing the PHOCNet

    :param t_phocnet: type of PHOCNet (normal or stn)
    :param alphabet: alphabet used
    :param phoc_lvls: levels of PHOC
    :return: specified PHOCNet
    """
    if t_phocnet == 'normal':
        phocnet = PHOCNet(n_out=phoc_util.len_phoc(levels=phoc_lvls, alphabet=alphabet))
    elif t_phocnet == 'stn':
        phocnet = STNPHOCNet(n_out=phoc_util.len_phoc(levels=phoc_lvls, alphabet=alphabet))
    else:
        raise ValueError('unknown PHOCNet type {}'.format(t_phocnet))
    return phocnet
