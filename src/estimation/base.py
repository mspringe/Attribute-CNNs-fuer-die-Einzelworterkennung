"""
In this module we introduce a base class for estimation, that is solely used for duck typing,
as well as a base class for an easy implementation of Metric based estimators.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
from copy import deepcopy
import warnings
from enum import Enum
import numpy as np
from scipy.spatial.distance import cdist
import json
import pickle
from src.util import sanity_util, phoc_util
from src.util.phoc_util import Alphabet


class Metrices(Enum):
    """
    Unique values for metrices

    Possible metrices are:

    * MAHALANOBIS
    * EUCLIDEAN
    * COSINE
    """
    MAHALANOBIS = 1
    EUCLIDEAN = 2
    COSINE = 3


class Estimator:
    """Abstract class for all estimators."""

    def __init__(self, words, train_data=None):
        """
        An estimator shall be initialized with a list of words (the lexicon).
        Training data is optional

        :param words: List of words/ lexicon
        :param train_data: Optional training data of atbitrary type
        """
        self.phoc = train_data
        self._words = words

    @property
    def words(self):
        """
        Mutation of the lexicon is unwanted and unneeded, unless explicitly demanded.
        Hence we prohibit direct references.
        """
        return deepcopy(self._words)

    def estimate(self, query: np.array):
        """
        This method shall estimate the nearest PHOC

        :param query: Representation of attributes (e.g. PHOC/ Neural Codes)
        :return: Nearest PHOC of lexicon (self._words) and its index in that order
        """
        measures = self.process_of_measure(np.array([query]), self.phoc)[0]
        idx = np.argmin(measures)
        return self.phoc[idx], idx

    def process_of_measure(self, X, compare):
        """
        The underlying process of measure.

        :param X: set of vector to measure comparison for
        :param compare: attribute vectors to compare to
        :return: values representing how much the i-th instance of X is alike any vector in compare
        """
        raise NotImplementedError('process_of_measure has not been implemented yet')

    def retrieval_list(self, query, vecs):
        """
        Calculating the retrieval list, accoring to the process of measure

        :param query: query
        :param vecs: attribute vecotrs
        :return: retrieval list indices
        """
        measures = self.process_of_measure(query, vecs)
        return np.argsort(measures, axis=1)

    def est_word(self, query: np.array):
        """
        This method shall be used when you desire to estimate the word of a single query.
        Use :meth:`src.estimation.base.Estimator.estimate_set` when processing large queries, as that method shall be
        optimised, regarding runtime.

        :param query: Representation of attributes (e.g. Attribute Vector/ Neural Codes)
        :return: Respective word of nearest PHOC, according to the estimator
        """
        phoc, idx = self.estimate(query)
        return self.words[idx]

    def save(self, dir, name='estimator'):
        """
        This method pikles the estimator.
        NOTE: Pre-existing data/ estimators will NOT be overwritten. Please clean up outdated estimators manually.

        :param dir: directory to pikle
        :param name: name of the file
        """
        dir = sanity_util.safe_dir_path(dir)
        file_name = sanity_util.unique_file_name(dir=dir, fn=name, suffix='.pkl')
        with open(file_name, 'wb') as f_out:
            pickle.dump(self, f_out)

    def estimate_set(self, X):
        """
        Estimation of all queries in set X.
        This method shall be implemented in all subclasses, using a wrapped C-code when possible, as it greatly affects
        the runtime of your evaluation.

        :param X: Set of estimated attribute representations (e.g. PHOC, Neural Codes), with shape (n_samples, dim_phoc)
        :return: Respective estimated words, with shape (n_samples,)
        """
        ests = [self.est_word(x) for x in X]
        return np.array(ests)

    @staticmethod
    def n_neighbour(attr_vec: np.ndarray, space: np.ndarray, metric: Metrices.COSINE):
        """
        Nearest neighbour search of an attribute vector in a matrix of vectors

        :param attr_vec: Representation of attributes (e.g. PHOC/ Neural Codes)
        :param space: Vector space
        :param metric: Enum :class:`src.estimation.base.Metrices` value of the metric used for distance
        :return: Nearest vector of the vector space and its respective index
        """
        # parsing enum to input string of cdist (using an enum, to ensure metrices are in the desired st of fucntions)
        if metric == Metrices.MAHALANOBIS:
            str_metric = 'mah'
        elif metric == Metrices.EUCLIDEAN:
            str_metric = 'euclidean'
        elif metric == Metrices.COSINE:
            str_metric = 'cosine'
        # default metric shall be the cosine distance
        else:
            str_metric = 'cosine'
        # usign scipys cdist for faster runtime
        dists = cdist([attr_vec], space, metric=str_metric)
        idx_nn = np.argmin(dists)
        return space[idx_nn], idx_nn


class DistEstimator(Estimator):
    """
    A DistEstimator shall perform a nn search, with respect to a metric.
    Distance estimators are of almost identical behaviour. Their only hyper-parameter is the metric used.
    """

    def __init__(self, words, metric, phoc_level=phoc_util.DEFAULT_PHOC_LEVELS,
                 alphabet=[Alphabet.ASCII_LOWER, Alphabet.ASCII_DIGITS, Alphabet.ASCII_PUNCTUATION], unambiguous=False):
        """
        Much like the generic Estimator, the distance estimator requires a lexicon of words, yet it operates on the
        respective PHOC and requires no training data.

        :param words: List of words/ lexicon
        :param metric: Metric used for nn search
        :param phoc_level: Number of levels in PHOC
        :param alphabet: Alphabet used for PHOC
        :param unambiguous: PHOC can be identical, this parameter indicates the discarding of ambbiguous PHOC in a
                            deterministic manner
        """
        self.__phoc_level = phoc_level
        self.__alphabet = alphabet
        self.__words = []
        self.phoc = []
        self.ambiguous = []
        self._metric = metric
        self.unambiguous = unambiguous
        # CAUTION setting word requires other global attributes to be set already
        self.words = words

    @property
    def words(self):
        return deepcopy(self.__words)

    @words.setter
    def words(self, new_words):
        """
        Setting the lexicon requires calculating the respective PHOC.
        As this estimator is solely based on the distances to the lexicons PHOC, we have to ensure their sanity
        and warn, if impurities should occur.

        .. note::

            self.ambiguous contains problematic words.

        :param new_words: New lexicon
        """
        # updating the lexicon
        self.__words = np.array(list(set(new_words)))
        # updating the PHOC
        self.phoc = [phoc_util.phoc(word=w, alphabet=self.__alphabet, levels=self.__phoc_level)
                      for w in self.__words]
        # eliminating zero PHOC vectors (those would inherently be the nearest neighbour for the cosine distance and we
        # we shall only  consider words that we can generate a representation for)
        self.phoc = np.array(self.phoc)
        sums = self.phoc.sum(axis=1)
        if any(sums == 0):
            warnings.warn('{} zero phocs\n{}'.format((sums == 0).sum(), self.__words[sums==0]))
        self.phoc = self.phoc[sums > 0]
        self.__words = self.__words[sums > 0]
        # checking for ambiguous PHOC and warn
        same_taken = np.zeros(self.phoc.shape[0], dtype=int)
        same_pairs = []
        same = 0
        for i in range(len(self.phoc)-1):
            zs = np.zeros(i + 1, dtype=int)
            tail = np.array(list(map(all, self.phoc[i] == self.phoc[i + 1:]))).astype(int)
            tmp_same = np.concatenate([zs, tail])
            tmp_same -= same_taken
            tmp_same[tmp_same < 0] = 0
            # pairs of words with identical PHOC
            if tmp_same.sum() > 0:
                same += 1
                same_pairs.append((self.__words[i], self.__words[tmp_same.astype(bool)]))
            same_taken += tmp_same
            same_taken[same_taken > 1] = 1
        # gathering ambiguous PHOC
        if same > 0:
            warnings.warn('{} same phocs out of {}\n{}'.format(same, len(self.phoc), same_pairs))
        # gathering ambiguous words, if this set is to large, you might want to use deeper PHOC (more levels)
        for w, pair in same_pairs:
            self.ambiguous.append(w)
            for v in pair:
                self.ambiguous.append(v)
        self.ambiguous = list(set(self.ambiguous))
        # discarding unambiguous PHOC if desired
        if self.unambiguous:
            self.__words = self.__words[~same_taken.astype(bool)]
            self.phoc = self.phoc[~same_taken.astype(bool)]

    def dists(self, query: np.ndarray):
        """
        Wrapping the cdist function, to ensure it is called correctly, with respect to this estimators metric

        :param query: query of attribute representations
        :return: distances to the lexicons respective PHOC
        """
        return self.process_of_measure(query, self.phoc)

    def process_of_measure(self, X, compare):
        return cdist(X, compare, metric=self._metric)

    def estimate(self, query: np.array):
        # distances to all queries
        dists = self.dists(query)
        # nearest neighbour
        argmin = np.argmin(dists)
        return dists[0][argmin], argmin

    def estimate_set(self, X):
        # calculating distances to all of the lexicons respective PHOC
        dists = self.dists(X)
        # nearest neighbours, i.e. minima per row
        idcs = np.argmin(dists, axis=1)
        # gather estimated words
        W_est = [self.__words[idx] for idx in idcs]
        return np.array(W_est)
