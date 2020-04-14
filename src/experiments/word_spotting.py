"""
This module provides a script to perform word spotting with a PHOCNet.
Word recognition and word spotting are closely related and the PHOCNet was created with word spotting in mind.

Example:

::

    python3 src/experiments/word_spotting.py \\
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
# libraries

from torch.utils.data import Dataset
import argparse
import json
import time
import os
import sys
import numpy as np
import pickle
# pytorch related imports
from torch.utils.data.dataloader import DataLoader
import torch
# own code
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.join(FILE_DIR, '..', '..', ''))
sys.path.append(SRC_DIR)
from src.nn.phocnet import PHOCNet
from src.io import dataloader
from src.util import eval_util, sanity_util, phoc_util, augmentation_util
from src.estimation import base, cosine
from src.parser.args_parser import parser_inference as parser
from src.parser.to_data import *
from src.io.dataloader import DSetPhoc


def attr_vecs(net: PHOCNet, dset: DSetPhoc, device=None, n_codes_lvl=0):
    """
    Calculates the attribute vectors

    :param net: A PHOCNet model
    :param dset: Dataset to gather attribute vectors and PHOC for
    :return: Attribute vector for images
    """
    # moving PHOCNet to specified GPU
    if device is not None:
        net.to(device)
    # collecting data
    vecs = []
    phocs = []
    bboxs= []
    forms = []
    transcripts = []

    d_loader = DataLoader(dset)
    for idx, data in enumerate(d_loader):
        ## forms
        forms.append(data['form'][0])
        ## PHOC
        phocs.append(data['phoc'][0].numpy())
        ## bounding box
        bboxs.append(np.array(data['bbox']).T)
        ## transcriptions
        transcripts.append(data['transcript'][0])
        ## attr. vector
        img = data['img']
        # retrieve estimated attribute vector
        batch_float = torch.tensor([img.numpy()], dtype=torch.float32)
        # move batch to GPU
        if device is not None:
            batch_float = batch_float.to(device)
        # storing attribute vector and PHOC
        v_attr = net.neural_codes(batch_float, pos=n_codes_lvl)
        v_attr = v_attr.cpu()
        v_attr = v_attr.detach().numpy()
        vecs.append(v_attr)
        # freeing some GPU memory
        batch_float.cpu()
    ## the word groups
    word_idcs = []
    taken = np.zeros(len(transcripts), dtype=bool)
    for k, w1 in enumerate(transcripts):
        w1_idcs = []
        for idx, w2 in enumerate(transcripts):
            if not taken[idx] and w1 == w2:
                w1_idcs.append(idx)
                taken[idx] = True
        if w1_idcs != []:
            word_idcs.append(w1_idcs)
    flattend = []
    for itms in word_idcs:
        flattend += itms
    print(len(dset), len(flattend))
    # moving PHOCNet back to CPU
    net.cpu()
    ## numpy conversions
    vecs = np.array(vecs)
    phocs = np.array(phocs)
    bboxs = np.array(bboxs)
    forms = np.array(forms)
    transcripts = np.array(transcripts)
    return vecs, phocs, bboxs, forms, transcripts, word_idcs


def run_wordspotting(net: PHOCNet, dset: dataloader.DSetPhoc, estimator:base.Estimator, device=None, n_codes_lvl=0):
    """
    Evaluation of the model, with restpect to the MAP.

    :param net: A PHOCNet model
    :param dset: Dataset to evaluate on
    :param device: GPU deivce. If set to None, computation will happen on the CPU. None by default.
    :return: MAP of the model on the dataset
    """
    dset = dset.sub_set(np.random.choice(len(dset), min(len(dset), 2000)))
    ## collecting data for evaluation
    net.eval()
    #TODO: collect these in a single for loop
    # array off attribute vectors
    v_attr_arr, phoc_arr, bbox_arr, form_arr, word_arr, word_idcs = np.array(attr_vecs(net=net, dset=dset, device=device,
                                                                                       n_codes_lvl=n_codes_lvl))
    ## evaluating MAP
    # list of binary entries indicating relevance for the respective item in the retrival list, regarding the query
    relevance_list = []
    # number of word-occurrences for the respective relance_list entry
    n_occ_list = []
    # filling lists
    w_err = 0
    w_err2 = 0
    for k, word_idcs in enumerate(word_idcs):
        # relevant word occurences
        bbox_gt = bbox_arr[word_idcs]
        forms_gt = form_arr[word_idcs]
        # collect binary relevances for each occurence of the word
        for idx in word_idcs:
            ret_list_idcs = estimator.retrieval_list(v_attr_arr[idx], estimator.phoc)[0] # eval_util.ret_list_idcs(v_attr_arr[idx][0], phoc_arr, metric='cosine')
            print(ret_list_idcs)
            w_err += 1 if word_arr[idx] != word_arr[ret_list_idcs[0]] else 0
            relevance, found_list = eval_util.relevance(arr_bbox_est=bbox_arr[ret_list_idcs], arr_bbox_gt=bbox_gt,
                                                        arr_form_est=form_arr[ret_list_idcs], arr_form_gt=forms_gt)
            relevance_list.append(relevance)
            n_occ_list.append(len(word_idcs))
        print(w_err, w_err/(k+1))
    # calculating the MAP
    MAP = eval_util.map(bin_relevances=relevance_list, occs=n_occ_list)
    return MAP


def load_net(nn_path):
    """loads net"""
    model = PHOCNet(n_out=phoc_util.len_phoc(levels=phoc_util.DEFAULT_PHOC_LEVELS))
    state_dict = torch.load(nn_path, map_location='cpu')
    model.load_state_dict(state_dict=state_dict)
    model.eval()
    return model


def write_map(map, dir_out, net_path, dset_name, f_name, estimator):
    """
    Writes the MAP to a JSON file

    :param map: MAP
    :param dir_out: Directory to write the output to
    :param net_path: Path of the PHOCNets states
    :param dset_name: Name of the dataset, e.g.: iam, gw, rimes
    """
    # creating save path
    t = time.asctime()
    sanity_util.safe_dir_path(dir_out)
    file_name = sanity_util.unique_file_name(dir=dir_out, fn=f_name, suffix='.json')
    # writing as JSON
    json_dict = {}
    json_dict['map'] = float(map)
    json_dict['dset_name'] = str(dset_name)
    json_dict['model'] = str(net_path)
    json_dict['estimator'] = str(type(estimator))
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
    n_codes_lvl = int(args['n_codes_lvl'])
    ## loading datasets
    data_set, train, test = get_dsets(dset_name, dset_csv, dset_imgs, alphabet, scale, k_fold=k_fold)
    # lexicon words
    words = list(sorted(set(train.words).union(set(test.words))))
    # using lowercase when needed
    test.lower_case = test.needs_lower(alphabet=alphabet)
    ## loading the estimator
    estimator = get_estimator(est_name, words, alphabet)
    ## initializing PHOCNet
    state_dict = torch.load(net_path, map_location='cpu')
    phocnet = get_PHOCNet(t_phocnet, alphabet)
    phocnet.load_state_dict(state_dict)
    ## calculating the MAP
    map = run_wordspotting(phocnet, test, estimator, device=device, n_codes_lvl=n_codes_lvl)
    # writing error rates
    write_map(map=map, dir_out=dir_out, net_path=net_path, dset_name=dset_name, f_name=file_name, estimator=estimator)
