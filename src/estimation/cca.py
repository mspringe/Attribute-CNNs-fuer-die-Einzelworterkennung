"""
This module provides a class to carry out the regularised CCA, lexicon based approach.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
# extending base libs
from enum import Enum
from copy import deepcopy
import numpy as np
import json
# from sklearn.cross_decomposition import CCA
# own libs
from src.estimation.base import Estimator, Metrices
from src.util import phoc_util, sanity_util
from src.util.phoc_util import Alphabet
from src.pyrcca.rcca import CCA as RCCA
from sklearn.preprocessing import normalize # USE!!!
from scipy.spatial.distance import cdist
import warnings


class RCCAEstimator(Estimator):
    """
    The RCCAEstimator performs a regularized CCA and a nearest neighbour search on the transformed data

    .. note::
        The RCCAEstimator additionally logs its configuration in a json file, when saved
    """

    def __init__(self, words,
                 n_dim=128, reg=10e3,
                 metric=Metrices.COSINE, phoc_lvls=phoc_util.DEFAULT_PHOC_LEVELS,
                 alphabet=[Alphabet.ASCII_LOWER, Alphabet.ASCII_DIGITS, Alphabet.ASCII_PUNCTUATION]):
        """
        The two main hyper parameters are the number of dimensions (n_dims) of the output vectors and the regularization
        paramter (reg)

        :param words: Lexicon of words
        :param n_dim: Number of dimensions for the output vector
        :param reg: Regularization parameter, used to avoid singularity of matrices
        :param metric: Metric to be used in the subspace, cosine distance per default, as no other makes obvious sense
        :param phoc_lvls: Number of Levels in PHOC
        :param alphabet: Alphabet used in PHOC
        """
        super().__init__([], [])
        # globals
        self.__alphabet = alphabet
        self.__phoc_level = phoc_lvls
        self.phoc = None
        self.phoc_trans = None
        self.weights_X = None
        self.weights_PHOC = None
        self.metric = metric
        # regularized CCA
        self.cca = RCCA(numCC=n_dim, reg=reg, verbose=False, kernelcca=False) # DO NOT use kernel cca
        self.reg = reg
        self.n_dim = n_dim
        # setting lexicon property (initialization of words list and respective PHOC)
        self.words = words

    @property
    def words(self):
        return deepcopy(self._words)

    @words.setter
    def words(self, words):
        """setting this property refreshes the train-data/ list of respecive PHOC aswell"""
        self._words = list(words)
        self.phoc = [phoc_util.phoc(word=w, alphabet=self.__alphabet, levels=self.__phoc_level).astype(float)
                     for w in self._words]
        self.phoc = np.array(self.phoc)

    def norm(self, X):
        """
        Method used for normalization, CCA demands zero mean and unit variance of all datasets

        :param X: Dataset of samples
        :return: Normalized dataset
        """
        return normalize(X, axis=1)

    def fit(self, X, Y, normalize=True):
        """
        Training the regularized CCA

        :param X: array-like of e.g. neural Codes
        :param Y: array-like of respective PHOC
        """
        # sanity
        X, Y = map(sanity_util.np_arr, [X, Y])
        # normalization
        if normalize:
            X, Y = map(self.norm, [X, Y])
        # training regularized CCA
        vdata = [X, Y]
        #vdata = np.array(vdata)
        self.cca.train(vdata)
        # weights of bases (used to transform into subspace)
        self.weights_X, self.weights_PHOC = deepcopy(self.cca.ws)
        # transforming PHOC of lexicon
        _, self.phoc_trans = self.transform(X, self.phoc)

    def transform(self, X, Y, normalize=True):
        """
        Implementation of the missing transform method in pyrcca

        :param X: Set of test samples of the X dataset
        :param Y: Set of test samples of the Y dataset
        :param normalize: Indicates whether to apply normalization after estimation
        :return: Transformed X and Y datasets in subspace
        """
        # sanity
        X, Y = map(sanity_util.np_arr, [X, Y])
        # normalization
        if normalize:
            X, Y = map(self.norm, [X, Y])
        # transformation
        transformed = [X.dot(self.weights_X), Y.dot(self.weights_PHOC)]
        # final normalization
        if normalize:
            X_trans, Y_trans = map(self.norm, transformed)
            #todo
        else:
            X_trans, Y_trans = transformed
        return X_trans, Y_trans

    def estimate_set(self, X, normalize=True):
        """
        Estimation of an entire Set. This should work better, than estimating samples individually, due to disparity in
        the normalization of the attributes.
        The dataset X and the PHOC of the lexicon will be transformed into the subspace in batches of the same size, to
        have simmilarly normalization behaviour.

        :param X: Queries, to be estimated
        :param normalize: Indicates whether to apply normalization after estimation
        :return: List of estimated words
        """
        dists = self.process_of_measure(X, self.phoc, normalize=normalize)
        idcs = np.argmin(dists, axis=1)
        # using a local variable, as self.words is treated as a function and would create loads of deep copies otherwise
        words = self.words
        return [words[idx] for idx in idcs]

    def process_of_measure(self, X, compare, normalize=True):
        # sanitize X set
        X = sanity_util.np_arr(X)
        X_trans, phoc_space = self.transform(X, compare, normalize=normalize)
        # nearest neighbour search in subspace
        if self.metric == Metrices.MAHALANOBIS:
            str_metric = 'mah'
        elif self.metric == Metrices.EUCLIDEAN:
            str_metric = 'euclidean'
        elif self.metric == Metrices.COSINE:
            str_metric = 'cosine'
        else:
            str_metric = 'cosine'
        dists = cdist(X_trans, phoc_space, metric=str_metric)
        return dists

    def nn_search_idcs(self, X, Y, metric=Metrices.COSINE):
        """
        Searching for the nearest neighbours of X in Y

        :param X: dataset to search nearest neighbours for
        :param Y: dataset to search for nearest neighbours in
        :param metric: metric to be used (see :class:`Metrices`)
        :return: a list of indices for nearest neighbours and a list of respective distances
        """
        if metric == Metrices.MAHALANOBIS:
            str_metric = 'mah'
        elif metric == Metrices.EUCLIDEAN:
            str_metric = 'euclidean'
        elif metric == Metrices.COSINE:
            str_metric = 'cosine'
        else:
            str_metric = 'cosine'
        # flattened distances
        dists = cdist(X, Y, metric=str_metric)
        # minium per row
        idcs = np.argmin(dists, axis=1)
        return idcs, dists[np.arange(len(dists)), idcs]

    def save(self, dir, name='estimator'):
        super().save(dir, name)
        # additionally keeping track of configuration
        file_config = sanity_util.unique_file_name(dir, name, '.json')
        with open(file_config, 'w') as f_config:
            json.dump({'reg': self.reg, 'n_dim': self.n_dim}, f_config)

