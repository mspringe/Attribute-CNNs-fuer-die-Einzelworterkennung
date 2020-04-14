"""
This script is used to run over a list of PHOCNets and evaluate them.
Their evaluations will be written to a json file and they will be processed according to the sorted list of their names.

Example:

::

    python3 visualize_nn_progress.py path/to/state_dict path/to/dir_out dset_name path/to/dset_annotations path/to/imgs --gpu_idx=cuda:0


For options have a look at :func:`src.parser.args_parser.parser_inference`


.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
import argparse
import numpy as np
import json
import pickle
import torch
from torch.utils.data.dataloader import DataLoader
import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.join(FILE_DIR, '..', '..', ''))
sys.path.append(SRC_DIR)
from src.io.dataloader import *
from src.estimation import cosine, cca, prob, base
from src.nn.phocnet import PHOCNet
from src.util import sanity_util, phoc_util
from src.training.phocnet_trainer import new_logger
from src.parser.args_parser import parser_inference as parser
from src.parser.to_data import *
from src.experiments.word_rec import run_word_rec


def evaluate_dir(dir, device, dset_test, estimator: base.Estimator, phocnet: PHOCNet, logger, s_batch=1, n_codes_lvl=0):
    """
    Evaluation of all model states in a directory

    :param dir: Directory, where the states of the individual models are located
    :param device: GPU device, e.g. cuda:0
    :param dset_test: Test set
    :param estimator: Estimator, to use for evaluation e.g. :class:`src.estimation.cosine.CosineEstimator`
    :param phocnet: PHOCNet instance to be initialized with the models states
    :param logger: Logger, to keep track of progress
    :param s_batch: batch-size to process in inference
    :return: Word errors and character arrors in that order
    """
    # processing the models states in alphabetical order
    nets = sorted(map(lambda p: os.path.join(dir, p), [net for net in os.listdir(dir) if net.endswith('.pth')]))
    # keeping track of word and character errors
    results = []
    # processing all model states
    for net_path in nets:
        # loading the state
        state_dict = torch.load(net_path, map_location='cpu')
        phocnet.load_state_dict(state_dict=state_dict)
        # evaluating the PHOCNet
        errs = run_word_rec(phocnet, dset_test, estimator, device=device, n_codes_lvl=n_codes_lvl)
        # storing the results
        results.append((net_path, errs))
        # moving th network back to the CPU as soon as possible
        phocnet.cpu()
        # logging latest result
        logger.info(f'{net_path}:\n\t{errs}\n')
    return results


def plot_series(errs, dir_out, name):
    """
    Plotting the mean word and character errors

    :param errs: tuples of PHOCNet paths and respective errors
    :param dir_out: output directory
    :param name: plot base-title
    """
    plot_w_err_path = sanity_util.unique_file_name(dir_out, name + '_w_err', '.png')
    plot_c_err_path = sanity_util.unique_file_name(dir_out, name + '_c_err', '.png')
    # gathering data
    x_ticks = []
    w_errs = []
    c_errs = []
    for n_path, e_dict in errs:
        n_name = os.path.basename(n_path)
        x_ticks.append(n_name)
        w_errs.append(e_dict['mean_w_err'])
        c_errs.append(e_dict['mean_c_err']['mean_pct'])
    # plotting word error
    plt.plot(np.arange(len(w_errs)), w_errs)
    plt.xticks(np.arange(len(w_errs)), x_ticks, rotation=25, rotation_mode="anchor",
               horizontalalignment='right', verticalalignment='top')
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.gca().set_ylabel('WER')
    plt.savefig(plot_w_err_path)
    plt.close(fig)
    plt.clf()
    # plotting character error
    plt.plot(np.arange(len(c_errs)), c_errs)
    plt.xticks(np.arange(len(c_errs)), x_ticks, rotation=25, rotation_mode="anchor",
               horizontalalignment='right', verticalalignment='top')
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.gca().set_ylabel('CER')
    plt.savefig(plot_c_err_path)
    plt.close(fig)
    plt.clf()


if __name__ == '__main__':
    ## argsparsing
    parser = parser()
    args = vars(parser.parse_args())
    # establish dset
    name_dset = args['dset_name']
    gt_path = args['dset_csv']
    imgs_path = args['dset_src']
    s_batch = int(float(args['s_batch']))
    # establish input and output dimensions
    alphabet = args['alphabet']
    alphabet = phoc_util.rep_to_alphabet(alphabet)
    scale_str = args['scale_w'], args['scale_h']
    scale = []
    for x in scale_str:
        try:
            scale.append(int(x))
        except Exception:
            scale.append(None)
    str_est = args['estimator']
    dir = args['net_path']
    str_device = str(args['gpu_idx'])
    if not str_device.startswith('cuda:'):
        device = None
    else:
        device = torch.device(str_device)
    dir_out = args['dir_out']
    dir_out = sanity_util.safe_dir_path(dir_out)
    name = args['model_name']
    t_phocnet = args['PHOCNet_type'].lower()
    k_fold = int(args['k_fold'])
    n_codes_lvl = int(args['n_codes_lvl'])
    ## loading dataset
    dset, train, test = get_dsets(name_dset, gt_path, imgs_path, alphabet, scale, k_fold)
    words = list(set(train.words).union(test.words))
    ## estimator
    estimator = get_estimator(str_est, words,alphabet)
    ## loading PHOCNet
    phocnet = get_PHOCNet(t_phocnet, alphabet)
    ## collect error rates
    logger = new_logger(dir_out, name)
    errs = evaluate_dir(dir, device, test, estimator, phocnet, logger, s_batch=s_batch, n_codes_lvl=n_codes_lvl)
    # save error rates for plotting
    file_path = sanity_util.unique_file_name(dir_out, name, '.json')
    with open(file_path, 'w') as f_out:
        json.dump(errs, f_out)
    # plotting and savong the plot
    plot_series(errs, dir_out, name)
