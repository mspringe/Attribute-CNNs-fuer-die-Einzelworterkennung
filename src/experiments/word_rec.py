"""
This module provides a script to evaluate the WER and CER of a model.

Example:

::

    python3 src/experiments/word_rec.py \\
    path/to/state_dict \\
    path/to/dir_out \\
    dset_name \\
    path/to/dset_annotations \\
    path/to/imgs \\
    --gpu_idx=cuda:0 \\
    --estimator=cosine

For options have a look at :func:`src.parser.args_parser.parser_inference`

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
# base libs
import argparse
import os
import sys
import json
import pickle
import numpy as np
import string
# torch
import torch
from torch.utils.data import DataLoader
# own code
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.join(FILE_DIR, '..', '..', ''))
sys.path.append(SRC_DIR)
from src.nn.phocnet import PHOCNet, STNPHOCNet
from src.io import dataloader
from src.util import phoc_util, sanity_util,augmentation_util
from src.estimation import base, cosine, prob, cca, euclidean
from src.parser.args_parser import parser_inference as parser
from src.parser.to_data import *


def run_word_rec(net : PHOCNet, dset : dataloader.DSetPhoc, estimator : base.Estimator, device=None, n_codes_lvl=0,
                 debug=False):
    """
    This method performs the main word recognition and returns error rates in a JSON serilizable dictionary

    :param net: PHOCNet instance
    :param dset: Test Data to evaluate on
    :param estimator: Estimator, used for estimation
    :param device: GPU device
    :param n_codes_lvl: Level to extract neural codes from, 0 is equivalent to a normal forward pass
    :param debug: There will be no output written, but a sample of up to 400 estimated words and respective
                  transcriptions printed, if debug is set to True
    :param scale: Scale of images
    :return: A dictionary containing the tatoal character and word erros, as well as the means/ proportional errors.
    """
    list_v_attr = []
    list_trans = []
    # moving net to device
    if device is not None:
        net = net.to(device)
    net.eval()
    # evaluaring dataset
    # dset = dset.exclude_words(string.punctuation)
    d_loader = DataLoader(dset)
    if debug:
        c_debug = 0
    for data in d_loader:
        if debug:
            if c_debug > 40:
                break
            c_debug += 1
        # gather relevant data
        transcription = data['transcript']
        while isinstance(transcription, list):
            transcription = transcription[0]
        img = data['img']
        batch = torch.tensor([img.numpy()], dtype=torch.float32)
        # move tensor to gpu
        if device is not None:
            batch = batch.to(device)
        # estimate attribute vector
        v_attr = net.neural_codes(batch, pos=n_codes_lvl)
        v_attr = v_attr.cpu()
        v_attr = v_attr.detach().numpy()
        # gather values
        v_attr = v_attr[0].astype(float)
        list_v_attr.append(v_attr)
        list_trans.append(transcription)
        # freeing GPU memory
        batch.cpu()
        del batch
    # freeing GPU memory
    net.cpu()
    # validation
    mat_v_attr = np.array(list_v_attr)
    est_words = estimator.estimate_set(mat_v_attr)
    w_err = np.sum([phoc_util.word_err(word=t, estimate=w) for t, w in zip(list_trans, est_words)])
    c_err = np.sum([phoc_util.char_err(word=t, estimate=w) for t, w in zip(list_trans, est_words)], axis=0)
    # when debugging is enabled we will only print the words and NOT write any files
    if debug:
        print('\n'.join(['transcription: {},{}est: {}'.format(t, ' '*(12-len(t)), e)
                         for t, e in zip(list_trans[:40], est_words[:40])]))
        raise Exception('Debugging enabled: ending after printed samples')
    # calculating means
    mean_w_err = w_err / len(dset)
    mean_c_err = c_err / len(dset)
    return {'w_err': int(w_err),
            'mean_w_err': float(mean_w_err),
            'c_err': {key : val
                      for key, val in zip(['total', 'summed_pcts'],
                                          list(c_err.astype(float)))},
            'mean_c_err': {key : val
                           for key, val in zip(['mean_total', 'mean_pct'],
                                               list(mean_c_err.astype(float)))}}


def save(dir_out, json_dict, name):
    """
    This method handles saving the errors to json files

    :param dir_out: Directory to save output file at
    :param json_dict: JSON object/ dictionary containing the errors
    :param name: The output files name
    """
    # safe path
    sanity_util.safe_dir_path(dir_out)
    file_name = sanity_util.unique_file_name(dir=dir_out, fn='{}_ERR'.format(name), suffix='.json')
    # writing JSON file
    with open(file_name, 'w') as f_json:
        json.dump(json_dict, f_json)


if __name__ == '__main__':
    ## args parsing
    parser = parser()
    args = vars(parser.parse_args())
    dir_out = args['dir_out']
    net_path = args['net_path']
    dset_name = args['dset_name']
    dset_csv = args['dset_csv']
    dset_imgs = args['dset_src']
    gpu_idx = args['gpu_idx']
    est_name = args['estimator']
    file_name = args['model_name']
    n_codes_lvl = int(args['n_codes_lvl'])
    k_fold = int(args['k_fold'])
    debug = False
    # checking for gpu device
    device = torch.device(gpu_idx) if gpu_idx != 'none' else None
    alphabet = phoc_util.rep_to_alphabet(args['alphabet'])
    lower_case = dataloader.Alphabet.ASCII_UPPER not in alphabet
    # scale of imgs
    scale_str = args['scale_w'], args['scale_h']
    scale = []
    for x in scale_str:
        try:
            scale.append(int(x))
        except Exception:
            scale.append(None)
    t_phocnet = args['PHOCNet_type'].lower()
    ## loading datasets
    data_set, train, test = get_dsets(dset_name, dset_csv, dset_imgs, alphabet, scale, k_fold)
    # lexicon words
    words = list(sorted(set(train.words).union(set(test.words))))
    # using lowercase when needed
    test.lower_case = test.needs_lower(alphabet=alphabet)
    if debug:
        print('data loaded')
    ## loading the estimator
    estimator = get_estimator(est_name, words, alphabet)
    ## initializing PHOCNet
    state_dict = torch.load(net_path, map_location='cpu')
    phocnet = get_PHOCNet(t_phocnet, alphabet)
    phocnet.load_state_dict(state_dict)
    ## calculating error rates
    json_dict = run_word_rec(net=phocnet, dset=test, estimator=estimator, device=device, n_codes_lvl=n_codes_lvl,
                             debug=debug)
    json_dict['model_name'] = file_name
    json_dict['path'] = net_path
    ## saving error rates
    save(dir_out=dir_out, json_dict=json_dict, name=file_name)
