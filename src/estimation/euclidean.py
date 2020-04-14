"""
This module provides a euclidean estimator (nn search with euclidean distance).

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
from src.util import phoc_util
from src.util.phoc_util import Alphabet
from src.estimation.base import DistEstimator

class EuclideanEstimator(DistEstimator):
    """DistEstimator with euclidean distance as metric"""

    def __init__(self, words, phoc_level=phoc_util.DEFAULT_PHOC_LEVELS,
                 alphabet=[Alphabet.ASCII_LOWER, Alphabet.ASCII_DIGITS, Alphabet.ASCII_PUNCTUATION]):
        super().__init__(words=words, metric='euclidean', phoc_level=phoc_level, alphabet=alphabet)
