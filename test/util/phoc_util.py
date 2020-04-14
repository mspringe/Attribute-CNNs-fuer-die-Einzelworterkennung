"""
PHOCNet relevant testing
.. codeauthor:: Maximilian Springenberg <maximilian.springenberg@tu-dortmund.de>
"""
from unittest import TestCase

import string
import numpy as np

# from src.nn.gpp import GPP
from src.util import phoc_util


class TestPHOCNet(TestCase):

    def setUp(self):
        pass

    def test_alphabet_chars(self):
        alphabet = [phoc_util.Alphabet.ASCII_LOWER, phoc_util.Alphabet.ASCII_UPPER, phoc_util.Alphabet.ASCII_DIGITS,
                    phoc_util.Alphabet.ASCII_PUNCTUATION]
        chars = phoc_util.alphabet_chars(alphabet)
        self.assertEqual(chars, string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation)

    def test_phoc(self):
        alphabet = [phoc_util.Alphabet.ASCII_LOWER, phoc_util.Alphabet.ASCII_UPPER, phoc_util.Alphabet.ASCII_DIGITS,
                    phoc_util.Alphabet.ASCII_PUNCTUATION]
        chars = phoc_util.alphabet_chars(alphabet)
        levels = 2
        word = 'aAzZ19.,'
        # building PHOC manually
        phoc_2 = np.zeros(len(chars), dtype=np.uint8)
        for char in word:
            phoc_2[chars.index(char)] = 1
        phoc_1_1 = np.zeros(len(chars), dtype=np.uint8)
        for char in word[:int(len(word)/2)]:
            phoc_1_1[chars.index(char)] = 1
        phoc_1_2 = np.zeros(len(chars), dtype=np.uint8)
        for char in word[int(len(word)/2):]:
            phoc_1_2[chars.index(char)] = 1
        phoc = np.concatenate((phoc_2, phoc_1_1, phoc_1_2))
        # test
        test_phoc = phoc_util.phoc(word=word, alphabet=alphabet, levels=levels)
        self.assertEqual(phoc.dtype, test_phoc.dtype)
        np.testing.assert_array_equal(phoc, test_phoc)
