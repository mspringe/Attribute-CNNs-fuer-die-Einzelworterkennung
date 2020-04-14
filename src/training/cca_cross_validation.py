"""
This module provides methods to run a cross validation for an RCCA estimator.


Example use-case:

::

    python3 src/training/cca_cross_validation.py \\
    path/to/dir_out \\
    gw \\
    path/to/gw_database/almazan/queries/queries.gtp \\
    path/to/gw_database/almazan/images \\
    path/to/PHOCNet_statedict
    --k_fold=1 \\
    --model_name=my_RCCA \\
    --gpu_idx=cuda:0 \\
    --alphabet=ldp


.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
import sys
import os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.join(FILE_DIR, '..', '..', ''))
sys.path.append(SRC_DIR)
sys.path.append(FILE_DIR)
from src.training.cca_trainer import train_cca, gather_NC_PHOC_pairs, parser, new_logger
from src.io import dataloader
from src.nn.phocnet import PHOCNet
from src.util import phoc_util, augmentation_util
from src.experiments.word_rec import run_word_rec
from src.estimation.cca import RCCAEstimator
from src.parser.to_data import *
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np


def w_err_CCA(cca: RCCAEstimator, n_codes, transcripts):
    """
    Calculating the number of words, that have been misclassified

    :param cca: an RCCA estimator
    :param n_codes: neural codes/ estimated PHOC
    :param transcripts: true transcriptions for respective  neural codes
    :return: number of words, that have been misclassified
    """
    estimates = cca.estimate_set(n_codes)
    w_err = np.sum([phoc_util.word_err(w, w_est) for w, w_est in zip(transcripts, estimates)])
    return w_err


def gather_NC_TRANS_pairs(net, dset, logger, device, n_code_lvl=0, s_batch=1):
    """
    Gathering the neural codes/ estimated PHOC from a dataset.

    :param net: a PHOCNet
    :param dset: test dataset
    :param logger: logger of infromation
    :param device: gpu-device to place the PHOCNet on
    :param n_code_lvl: level to extract neural codes from (see :func:`src.nn.phocnet.PHOCNet.neural_codes`)
    :param s_batch: size of batch for inference
    :return: neural codes and true transcriptions in that order
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
    trans = []
    batch = []
    N = len(d_loader)
    for i, data in enumerate(d_loader):
        # gather image and respective PHOC
        img = data['img']
        transscript = data['transcript']
        while isinstance(transscript, list):
            transscript = transscript[0]
        # batch with size 1 for the image
        batch.append(img.numpy())
        trans.append(transscript)
        if i == (N-1) or (i+1) % s_batch == 0:
            batch = torch.tensor(batch, dtype=torch.float32)
            # moving to gpu
            if device is not None:
                batch = batch.to(device)
            # extract neural codes
            n_codes = net.neural_codes(batch, pos=n_code_lvl).cpu().detach().numpy().astype(float).tolist()
            # storing vectors
            neural_codes += n_codes
            # emptying batch
            batch = []
    logger.info('done gathering')
    # freeing the gpu memory
    net.cpu()
    # numpy conversion of samples
    neural_codes = np.array(neural_codes)
    trans = np.array(trans)
    return neural_codes, trans


def cca_run(dims, regs, dset_train, dset_test, net, words, device, logger, alphabet):
    """
    Cross validation of different hyper parameters for regularised CCA

    :param dims: dimension paraameters of the CCAs subspace
    :param regs: regularization parameters
    :param dset_train: training dataset
    :param dset_test: test dataset
    :param net: a PHOCNet
    :param words: lexicon of words
    :param device: gpu device to place the PHOCNet on
    :param logger: logger of information
    :param alphabet: alphabet for PHOC
    :return: word errors and respective configurations in that order
    """
    errs = []
    configs = []
    neural_codes, phocs = gather_NC_PHOC_pairs(net=net, dset=dset_train, logger=logger, device=device)
    neural_codes_val, trans = gather_NC_TRANS_pairs(net=net, dset=dset_test, logger=logger, device=device)
    # evaluation of configs
    for reg in regs:
        for dim in dims:
            # training a config
            cca = train_cca(neural_codes=neural_codes, phocs=phocs, dim=dim, reg=reg, logger=logger, words=words,
                            alphabet=alphabet)
            logger.info('trained config reg:{}, dim:{}'.format(reg, dim))
            # evaluating that config
            err = w_err_CCA(cca=cca, n_codes=neural_codes_val, transcripts=trans)
            logger.info('evaluated config reg:{}, dim:{} with WE:{}'.format(reg, dim, err))
            # storing data on config
            errs.append(err)
            configs.append((reg, dim))
    return np.array(errs), configs


def cross_val(dset_train: dataloader.DSetPhoc, net, words, device, logger, n_fold=4,
              alphabet=phoc_util.DEFAULT_ALPHABET):
    """
    Running cross validation with fixed hyperparameters.

    :param dset_train: training dataset
    :param net: a PHOCNet
    :param words: lexicon of words
    :param device: gpu device to place the PHOCNet on
    :param logger: logger of infomation
    :param n_fold: number of folds for the cross validatin
    :param alphabet: alphabet for HOC
    :return: the besst cca and the corresponding mean word error
    """
    regs = [10, 10e1, 10e2, 10e3, 10e4, 10e5, 10e6]
    dims = [16, 32, 64, 128, 256]
    idcs = np.arange(len(dset_train))
    folds = [ ( np.array(list(set(idcs).difference(idcs[i::n_fold]))), idcs[i::n_fold] ) for i in range(n_fold) ]
    total_errs = None
    for idx, (idcs_train, idcs_test) in enumerate(folds):
        train = dset_train.sub_set(idcs_train)
        test = dset_train.sub_set(idcs_test)
        errs, configs = cca_run(dims=dims, regs=regs, dset_train=train, dset_test=test, net=net, words=words,
                                device=device, logger=logger, alphabet=alphabet)
        if total_errs is None:
            total_errs = errs
        else:
            total_errs += errs
        log.info('ran fold {}/{}{}errors:{}'.format(idx+1, len(folds), ' ' * len('ran fold{}'.format(idx)),
                                                    list(zip(total_errs, configs))))
    # best results
    reg, dim = configs[np.argmin(total_errs)]
    logger.info('best result for reg:{}, dim:{}'.format(reg, dim))
    neural_codes, phocs = gather_NC_PHOC_pairs(net=net, dset=dset_train, logger=logger, device=device)
    cca = train_cca(neural_codes=neural_codes, phocs=phocs, dim=dim, reg=reg, logger=logger, words=words, alphabet=alphabet)
    return cca, (total_errs/n_fold)


if __name__ == '__main__':
    # args parsing
    parser = parser()
    args = vars(parser.parse_args())
    dset_name = args['dset_name']
    dset_csv = args['dset_csv']
    dset_imgs = args['dset_src']
    net_path = args['net_path']
    name = args['model_name']
    gpu_idx = args['gpu_idx']
    dir_out = args['dir_out']
    t_phocnet = args['PHOCNet_type']
    n_code_lvl = int(args['n_code_lvl'])
    k_fold = int(args['k_fold'])
    phoc_lvls = int(args['phoc_lvls'])
    augment = (args['augment'].lower() in ['true', '1', 'y', 'yes'])
    alphabet = phoc_util.rep_to_alphabet(args['alphabet'])
    lower_case = dataloader.Alphabet.ASCII_UPPER not in alphabet
    scale_str = args['scale_w'], args['scale_h']
    scale = []
    for x in scale_str:
        try:
            scale.append(int(x))
        except Exception:
            scale.append(None)
    # dataset
    _, data_set, test = get_dsets(dset_name, dset_csv, dset_imgs, alphabet, scale, k_fold, phoc_lvls)
    data_set = data_set.apply_alphabet(alphabet)

    # gpu device
    device = torch.device(gpu_idx) if gpu_idx != 'none' else None

    # initializing PHOCNet
    state_dict = torch.load(net_path, map_location='cpu')
    phocnet = get_PHOCNet(t_phocnet, alphabet, phoc_lvls)
    phocnet.load_state_dict(state_dict=state_dict)

    # running the cross validation
    log = new_logger(dir_out, name)
    est_cca, mean_errs = cross_val(dset_train=data_set, net=phocnet, words=sorted(data_set.words), device=device, logger=log,
                                   n_fold=4, alphabet=alphabet)
    # saving estimator
    est_cca.save(dir=dir_out, name=name)

    wc_err = run_word_rec(net=phocnet, dset=test, estimator=est_cca, device=device, n_codes_lvl=0,
                          debug=False)
    log.info('evaluated model with best CCA: {}'.format(wc_err))