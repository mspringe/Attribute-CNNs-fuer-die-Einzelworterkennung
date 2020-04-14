"""
Probabilistic approach, as proposed by Eugen Rusakov, et al.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
from copy import deepcopy

import numpy as np

from src.estimation import base
from src.util import phoc_util
from src.util.phoc_util import Alphabet

class ProbEstimator(base.Estimator):
    """
    This estimator uses the probabilistic retrieval model, as proposed by Rusakov, et al.

    .. note::
        as this is not a classic distance, we have to used python methods, hence this estimator will be slightly
        slower and estimate_set is implicitly defined by estimate
    """

    def __init__(self, words, phoc_level=phoc_util.DEFAULT_PHOC_LEVELS,
                 alphabet=[Alphabet.ASCII_LOWER, Alphabet.ASCII_DIGITS, Alphabet.ASCII_PUNCTUATION]):
        """
        tain_data is initialized with the PHOC encodings of the handed words
        words will be stored seperately

        :param words: words in dictionary
        :param phoc_level: levels of PHOC-encoding
        :param alphabet: alphabet used for PHOC (see :class:`phoc_util.Alphabet`)
        """
        self.__phoc_level = phoc_level
        self.__alphabet = alphabet
        super().__init__(words,
                         [phoc_util.phoc(word=w, alphabet=self.__alphabet, levels=self.__phoc_level)
                          for w in words])
        self.words = words

    @property
    def words(self):
        """The lexicon shall be immutable, unless it is explicitly set"""
        return deepcopy(self._words)

    @words.setter
    def words(self, new_words):
        """
        The PHOC have to updated with the lexicon

        :param new_words:  new lexicon
        """
        # updating PHOC-table
        self.train_data = [phoc_util.phoc(word=w, alphabet=self.__alphabet, levels=self.__phoc_level)
                           for w in new_words]
        self._words = new_words

    def prm_scores(self, est_phoc : np.ndarray):
        """
        calculating the PRM scores corresponding for an estimated PHOC

        :param est_phoc: estimated PHOC
        :return: PRM scores, with respect to the current lexicon
        """
        # all PRM-scores
        probs = [self.__posterior(phoc, est_phoc) for phoc in self.train_data]
        return probs

    def estimate(self, est_phoc : np.ndarray):
        """
        Estimation via highest PRM-score

        :param est_phoc: Probability of atributes (estimated PHOC of the Attribute-CNN)
        :return: String of estimated word
        """
        probs = self.prm_scores(est_phoc=est_phoc)
        # word with highest PRM-score
        idx_est = np.argmax(probs)
        v = self.train_data[idx_est]
        return v, idx_est

    def process_of_measure(self, X, compare):
        return np.array([[self.__posterior(x_i, c_j) for c_j in compare] for x_i in X])

    def __posterior(self, qa: np.array, est_a: np.array):
        """
        PRM-score as proposed in :ref:`paper` (IV)

        :param qa: PHOC encoding of query/ word in dictionary
        :param est_a: Probability of atributes (estimated PHOC of the Attribute-CNN)
        :return: PRM-score
        """
        # sanity checks
        if not isinstance(qa, np.ndarray):
            qa = np.array(qa)
        if not isinstance(est_a, np.ndarray):
            est_a = np.array(est_a)
        # stability of log (non zero input)
        eps = 1e-10
        # formula: Sum (qa_i * log(est_a) + (1-qa_i) * log(1-est_a))
        vals = qa * np.log(est_a+eps) + (1-qa) * np.log(1 - est_a+eps)
        prm_score = np.sum(vals)
        return prm_score
