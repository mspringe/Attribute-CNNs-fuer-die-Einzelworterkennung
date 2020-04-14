"""
Training script, used for the PHOCNet.

Output will be written to the directory specified as dir_out.
Each model has its own directory containing:

    * The log file
    * The final state-dictionary
    * A config JSON file of the arguments provided to the script
    * A config JSON file of PHOCNet (stating the PHOCNets configuration)
    * A directory "tmp" containing state-dictionaries, that have been saved during training


Example for training the PHOCNet:

::

    python3 src/training/phocnet_trainer.py \\
    path/to/output_dir/ \\
    gw \\
    path/to/gw_database/almazan/queries/queries.gtp \\
    path/to/gw_database/almazan/images \\
    --max_iter=1e5 \\
    --model_name=my_PHOCNet \\
    --gpu_idx=cuda:0 \\
    --k_fold=1 \\
    --alphabet=ldp \\
    --s_batch=10

See also :func:`src.parser.args_parser.parser_training` for all options, regarding training.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
# general libraries
import os
import logging
import string
import time
import pickle
import argparse
import sys
import json
# pytorch relevant imports
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch
# own libs
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.join(FILE_DIR, '..', '..', ''))
sys.path.append(SRC_DIR)
sys.path.append(FILE_DIR)
from src.nn import phocnet
from src.io.dataloader import *
from src.util import phoc_util, sanity_util
from src.util.phoc_util import Alphabet
from src.parser.args_parser import parser_training as parser
from src.parser.to_data import *


class Trainer:
    """generic trainer of models"""

    def __init__(self, net : phocnet.PHOCNet, net_log_dir, loss=nn.BCELoss(), s_batch=1, device=None,
                 logger=None, augmented=True, s_aug=500000, quant_aug=DSetQuant.EQUAL, tmp_save_mod=10000,
                 alphabet=[Alphabet.ASCII_LOWER, Alphabet.ASCII_DIGITS, Alphabet.ASCII_PUNCTUATION],
                 phoc_lvls=phoc_util.DEFAULT_PHOC_LEVELS, mixed_precision=False, FP=torch.float32):
        # globals
        self.alphabet = alphabet
        self.phoc_lvls = phoc_lvls
        self.net_log_dir = net_log_dir
        self.loss = loss
        self.s_batch = s_batch
        self.nn = net
        self.__device = None
        self.__logger = logger
        self.augmented = augmented
        self.aug_size = s_aug
        self.aug_quant = quant_aug
        self.tmp_save_mod = tmp_save_mod
        self.mixed_precision = mixed_precision
        # properties
        self.device = device
        self.FP = FP

    @property
    def device(self):
        """device the NN shall run on"""
        return self.__device

    @device.setter
    def device(self, device):
        """changing the device to run on"""
        # moving to gpu if a device is provided
        if device is not None:
            self.nn.to(device=device)
        # going back to cpu otherwise
        else:
            self.nn.to('cpu')
        self.__device = device

    def train_on(self, d_set: DSetPhoc, optimizer, n_iter=1e5, shuffle=True):
        """
        The training loop

        :param d_set: Dataset to run on
        :param optimizer: Optimizer (e.g. :class:`optim.SGD`, :class:`optim.Adam`)
        :param n_iter: Number of iterations to be run
        :param shuffle: Indicates whether data shall be shuffled each epoch, True by default
        """
        # applying PHOC-settings to the dataset
        d_set.alphabet = self.alphabet
        d_set.phoc_levels = self.phoc_lvls
        # applying augmentation settings to the dataset
        if self.augmented:
            log.info('initializing augmented training data')
            d_set = d_set.augment(size=self.aug_size, t_quant=self.aug_quant)
            log.info('done, training on dataset of size {}'.format(len(d_set)))
        # making sure the logging dir exists
        sanity_util.safe_dir_path(self.net_log_dir)
        # empty batch and respective embeddings (batches via tensor not feasible, since images may vary in scale)
        batch = []
        embeddings = []
        # keeping track of iterations
        iter = 0
        # training
        # keeping track of the mean error
        mean_batch_err = 0
        # running the epochs
        training = True
        epoch = 0
        while training:
            # initialize train-data as shuffled (updating iterator each epoch, to decrease likelihood of overfitting)
            d_loader = DataLoader(dataset=d_set, shuffle=shuffle)
            for idx, data in enumerate(d_loader):
                # collecting the batch
                img = data['img']
                emb = data['phoc']
                if len(batch) < self.s_batch and idx < (len(d_set) - 1):
                    batch.append(img)
                    embeddings.append(emb)
                    continue
                else:
                    # processing batch
                    mean_batch_err += self.train_on_batch(batch=batch, embeddings=embeddings)
                    # counting iterations
                    iter += 1
                    # emptying batch
                    batch = []
                    embeddings = []
                    # optimizer step
                    optimizer.step()
                    optimizer.zero_grad()
                # making sure not to exceed maximum iterations during epoch
                if iter >= n_iter:
                    break
                # logging
                # logging mean batch error every 1000 iterations
                if iter % 1000 == 0:
                    tmp_err = mean_batch_err / 1000 / self.s_batch
                    mean_batch_err = 0
                    if self.__logger is not None:
                        self.__logger.info('iteration {}/ {} ended, mean error {}'.format(iter, n_iter, tmp_err))
                    # saving net state-dict in intervals (default 10000 iterations)
                    if iter % self.tmp_save_mod == 0:
                        tmp_path = os.path.join(self.net_log_dir, 'epoch_{}_iter{}.pth'.format(epoch, iter))
                        torch.save(self.nn.state_dict(), tmp_path)
                        if self.__logger is not None:
                            self.__logger.info('iteration {}/ {} ended, wrote net to {}'.format(iter, n_iter, tmp_path))
                    # changing learning rate after 60,000 iterations (dividing by 10)
                    # or changing learning rate every 100,000 iterations starting with the 200,000th iteration
                    if (iter % 60000 == 0 and iter < 100000) or (iter % 200000 == 0): #or (iter >= 200000 and iter % 100000 == 0):
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] / 10
            # making sure not to exceed maximum iterations after epoch
            if iter >= n_iter:
                training = False
                break
            # logging after batch
            if self.__logger is not None:
                self.__logger.info('epoch {} ended'.format(epoch))
            epoch += 1

    def train_on_batch(self, batch, embeddings):
        """
        Performs forward and backwards propagation on a given batch

        :param batch: batch to be processed
        :param embeddings: respective embeddings
        :return: summed losses of the batch
        """
        err = 0
        # input images of individual sizes --> individual forward passes
        if not all([batch[0].size() == e.size() for e in batch]):
            for img, emb in zip(batch, embeddings):
                # conversion to correct tensor
                x_in = torch.tensor([img.numpy()], dtype=self.FP)
                y_emb = torch.tensor(emb.numpy(), dtype=self.FP)
                # auto-grading
                x_in = torch.autograd.Variable(x_in)
                y_emb = torch.autograd.Variable(y_emb)
                # moving to gpu
                if self.device is not None:
                    x_in = x_in.to(self.device)
                    y_emb = y_emb.to(self.device)
                # forward pass
                x_out = self.nn(x_in).type(self.FP)
                # calculating loss
                loss = self.loss(x_out, y_emb)
                # backward propagation
                loss.backward()
                # summing losses
                err += loss.item()
        # fixed input size --> faster forward pass, via one stacked tensor
        else:
            x_in = torch.tensor([b.numpy() for b in batch], dtype=self.FP)
            y_emb = torch.tensor([e.numpy()[0] for e in embeddings], dtype=self.FP)
            # auto-grading
            x_in = torch.autograd.Variable(x_in)
            y_emb = torch.autograd.Variable(y_emb)
            # moving to gpu
            if self.device is not None:
                x_in = x_in.to(self.device)
                y_emb = y_emb.to(self.device)
            # forward pass
            x_out = self.nn(x_in).type(self.FP)
            # calculating loss
            loss = self.loss(x_out, y_emb)
            # backward propagation
            loss.backward()
            # summing losses
            err += loss.sum().item()
        # moving to cpu
        x_in.cpu()
        y_emb.cpu()
        del x_in
        del y_emb
        return err

    def set_up(self):
        """dictionary with meta data of training"""
        t_loss_str = str(self.loss)
        augmeted = self.augmented
        s_augmented = self.aug_size
        quant_augmented = str(self.aug_quant)
        return {'f_loss': t_loss_str, 'augmentation': augmeted, 'augmentation_size': s_augmented,
                'augmentation_quantification': quant_augmented, 'nn_setup': self.nn.setup(),
                'alphabet': phoc_util.alphabet_to_rep(self.alphabet), 'phoc_lvls': self.phoc_lvls}

    def save(self, dir_out, train=None, test=None, pfx=''):
        """saving the NN, aswell as all relevant meta-data"""
        # creating save path
        sanity_util.safe_dir_path(dir_out)
        # not deleting prior data
        file_path = sanity_util.unique_file_name(dir=dir_out, fn='nn_{}'.format(pfx), suffix='.pth')
        file_path_setup = sanity_util.unique_file_name(dir=dir_out, fn='setup_{}'.format(pfx), suffix='.json')
        # writing nn
        torch.save(self.nn.state_dict(), file_path)
        # writing the training setup
        with open(file_path_setup, 'w') as f_json:
            json.dump(self.set_up(), f_json)


def sgd_optimizer(parameters, lr=0.01, momentum=0.9):
    """standard SGD optimizer"""
    optimizer = optim.SGD(parameters, lr=lr, momentum=momentum)
    return optimizer


def adam_optimizer(parameters, lr=0.0001):
    """adam optimizer as proposed in the Retsinas paper"""
    optimizer = optim.Adam(parameters, lr=lr, betas=(0.9, 0.99), weight_decay=0.00005)
    return optimizer


def new_logger(dir_out, name):
    """initializes a logger for training"""
    logger = logging.getLogger(name)
    dir_out = sanity_util.safe_dir_path(dir_out)
    log_file_path = sanity_util.unique_file_name(dir=dir_out, fn=name, suffix='.log')
    hdlr = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


class CosineLoss(nn.Module):
    """Cosine Loss has no native implementation in pytorch, hence this Module class (see :class:`nn.Module`)."""

    def __init__(self, size_average=True, use_sigmoid=False):
        super(CosineLoss, self).__init__()
        self.averaging = size_average
        self.use_sigmoid = use_sigmoid

    def forward(self, input_var, target_var):
        """
        calculates the cosine loss: :math:`1.0 - (y.x / |y|*|x|)`

        :param input_var: estimated vector
        :param target_var: embedding
        :return: cosine loss
        """
        if self.use_sigmoid:
            loss_val = sum(1.0 - nn.functional.cosine_similarity(torch.sigmoid(input_var), target_var))
        else:
            loss_val = sum(1.0 - nn.functional.cosine_similarity(input_var, target_var))
        if self.averaging:
            loss_val = loss_val/input_var.data.shape[0]
        return loss_val


if __name__ == '__main__':
    t_start_training = time.asctime()
    """arg parser"""
    arg_parser = parser()
    args = vars(arg_parser.parse_args())
    # FP16 vs FP32
    FP = torch.float32 #torch.float16 if int(args['FP']) == 16 else torch.float32
    # src dir of the dataset
    dset_csv = args['dset_csv']
    dset_src = args['dset_src']
    # fold number/ index
    k_fold = int(args['k_fold'])
    # scale of imgs
    scale_str = args['scale_w'], args['scale_h']
    scale = []
    for x in scale_str:
        try:
            scale.append(int(x))
        except Exception:
            scale.append(None)
    # figuring out which dataset to use and under which protocol
    phoc_lvls = int(args['phoc_lvls'])
    alphabet = phoc_util.rep_to_alphabet(args['alphabet'])
    stopwords = args['stop_words'].lower() in ['true', '1', 't', 'yes', 'y']
    punctuation = args['punctuation'].lower() in ['true', '1', 't', 'yes', 'y']
    dset_name = args['dset_name']
    data_set, train_set, test_set = get_dsets(dset_name, dset_csv, dset_src, alphabet, scale, k_fold, phoc_lvls)
    train_set = train_set.apply_alphabet(alphabet)
    # name of gpu device
    gpu_idx = args['gpu_idx']
    # number of maximum training epochs and iterations
    max_iter = int(float(args['max_iter']))
    # name of model
    model_pfx = args['model_name']
    # directory to save the NN at
    dir_out = os.path.join(args['dir_out'], model_pfx)
    # optimizer to be used
    optim_type_str = args['optimizer']
    # net type to be used
    net_type_str = args['PHOCNet_type']
    # loss function, used in training
    loss_str = args['loss']

    """initialize logging"""
    log = new_logger(dir_out=dir_out, name=model_pfx)

    """intialize the PHOCNet"""
    # choose module
    phoc_net = get_PHOCNet(net_type_str, alphabet, phoc_lvls)
    # weight initialization
    p_pretrained = args['pretrained']
    if os.path.isfile(p_pretrained):
        state_dict = torch.load(p_pretrained, map_location='cpu')
        # un-strict loading enables STNPHOCNet instances to be (partially) initialized with PHOCNet instances
        phoc_net.load_state_dict(state_dict=state_dict, strict=False)
    else:
        phoc_net.init_weights()

    """initialize optimizer"""
    lr = float(args['lr'])
    if optim_type_str == 'sgd':
        optimizer = sgd_optimizer(phoc_net.parameters(), lr=lr)
    else:
        optimizer = adam_optimizer(phoc_net.parameters(), lr=lr)

    """initialize trainer"""
    augment_dset = args['augment'].lower() in REP_STRS
    t_augment = rep_to_quant(args['augment'])
    intv_save = int(args['save_interval'])
    s_batch = int(args['s_batch'])
    device = torch.device(gpu_idx) if gpu_idx != 'none' else None
    if loss_str.lower() == 'bce':
        loss = nn.BCEWithLogitsLoss(size_average=True)
    else:
        loss = CosineLoss(size_average=False, use_sigmoid=False)
    trainer = Trainer(net=phoc_net, net_log_dir=os.path.join(dir_out, 'tmp', model_pfx, ''),
                      device=device, logger=log, loss=loss, s_batch=s_batch, augmented=augment_dset, tmp_save_mod=intv_save,
                      alphabet=alphabet, phoc_lvls=phoc_lvls, quant_aug=t_augment, FP=FP)

    """run training"""
    shuffle = args['shuffle'].lower() in ['true', '1', 't', 'yes', 'y']
    trainer.train_on(d_set=train_set, optimizer=optimizer, n_iter=max_iter)

    """save net"""
    ids_train = train_set.ids
    ids_test = test_set.ids
    trainer.save(dir_out=dir_out, train=ids_train, test=ids_test, pfx=model_pfx)
    """saving args, so you have a reference to the training-config of your model"""
    sanity_util.safe_dir_path(dir_out)
    file_path = sanity_util.unique_file_name(dir=dir_out, fn='args_{}'.format(model_pfx), suffix='.json')
    with open(file_path, 'w') as args_out:
        t_end_training = time.asctime()
        args['date'] = {'started': t_start_training, 'ended': t_end_training}
        json.dump(args, args_out)
