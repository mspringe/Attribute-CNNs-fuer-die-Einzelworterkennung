"""
This module provides a cosine estimator (nn search with cosine distance).

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
from src.util import phoc_util
from src.util.phoc_util import Alphabet
from src.estimation.base import DistEstimator

class CosineEstimator(DistEstimator):
    """DistEstimator with cosine distance as metric"""

    def __init__(self, words, phoc_level=phoc_util.DEFAULT_PHOC_LEVELS,
                 alphabet=[Alphabet.ASCII_LOWER, Alphabet.ASCII_DIGITS, Alphabet.ASCII_PUNCTUATION]):
        super().__init__(words=words, metric='cosine', phoc_level=phoc_level, alphabet=alphabet)
