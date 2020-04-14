"""
This module provides methods to generate deterministic PHOC encodings, as well as methods for the CER and WER.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
from copy import deepcopy
from enum import Enum
from typing import List

import numpy as np
import string


DEFAULT_PHOC_LEVELS = 3
ALPHABET_PERFECT_IAM = '!"#&\'()*+,-./0123456789:;?abcdefghijklmnopqrstuvwxyz'
ALPHABET_PERFECT_RIMES = "%'-/0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz°²"
ALPHABET_PERFECT_GW = '0123456789abcdefghijklmnopqrstuvwxyz'


class Alphabet(Enum):
    """"Attributes of an alphabet. Relevant to determine the semantic of a PHOC."""
    ASCII_LOWER = 1
    ASCII_UPPER = 2
    ASCII_PUNCTUATION = 3
    ASCII_DIGITS = 4
    PERFECT_IAM = 5
    PERFECT_RIMES = 6
    PERFECT_GW = 7


DEFAULT_ALPHABET = [Alphabet.ASCII_LOWER, Alphabet.ASCII_UPPER, Alphabet.ASCII_DIGITS]


def rep_to_alphabet(alph_chars: str):
    """
    way of parsing an alphabet from a string

    ================= =======
    alphabet property char
    ================= =======
    ASCII_LOWER       l
    ASCII_UPPER       u
    ASCII_PUNCTUATION p
    ASCII_DIGITS      d
    ================= =======

    :param alph_chars: characters indicating the alphabet attributes
    :return: list of alphabet attributes (see :class:`Alphabet`)
    """
    # perfect fits
    if np.sum([a in alph_chars for a in ['gw', 'iam', 'rimes']]) > 1:
        raise ValueError('multiple perfect fitting alphabets defeat purpose')
    if 'gw' in alph_chars:
        return [Alphabet.PERFECT_GW]
    if 'iam' in alph_chars:
        return [Alphabet.PERFECT_IAM]
    if 'rimes' in alph_chars:
        return [Alphabet.PERFECT_RIMES]
    # universal alphabets
    alph_chars = alph_chars.lower()
    mapping = {'l': Alphabet.ASCII_LOWER,
               'u': Alphabet.ASCII_UPPER,
               'p': Alphabet.ASCII_PUNCTUATION,
               'd': Alphabet.ASCII_DIGITS}
    alphabet = set()
    for c in alph_chars:
        alphabet.add(mapping[c])
    alphabet = list(alphabet)
    return alphabet


def alphabet_to_rep(alphabet: List[Alphabet]):
    """
    way of parsing an alphabet to a string

    ================= =======
    alphabet property char
    ================= =======
    ASCII_LOWER       l
    ASCII_UPPER       u
    ASCII_PUNCTUATION p
    ASCII_DIGITS      d
    ================= =======

    :param alphabet: properties of the alphabet (see :class:`Alphabet`)
    :return: string of characters representing alphabet properties
    """
    # perfect fits
    if np.sum([a in alphabet for a in [Alphabet.PERFECT_GW, Alphabet.PERFECT_IAM, Alphabet.PERFECT_RIMES]]) > 1:
        raise ValueError('multiple perfect fitting alphabets defeat purpose')
    if Alphabet.PERFECT_GW in alphabet:
        return 'gw'
    if Alphabet.PERFECT_IAM in alphabet:
        return 'iam'
    if Alphabet.PERFECT_RIMES in alphabet:
        return 'rimes'
    # universal alphabets
    mapping = {Alphabet.ASCII_LOWER: 'l',
               Alphabet.ASCII_UPPER: 'u',
               Alphabet.ASCII_PUNCTUATION: 'p',
               Alphabet.ASCII_DIGITS: 'd'}
    rep = set()
    for a in alphabet:
        rep.add(mapping[a])
    rep = list(rep)
    return rep


def alphabet_chars(alphabet: List[Alphabet]):
    """
    maps the alphabet-type to a list of strings

    :param alphabet: alphabet-type (see :class:`Alphabet`)
    :return: list of characters in alphabet-type
    """
    # perfect fits
    if np.sum([a in alphabet for a in [Alphabet.PERFECT_GW, Alphabet.PERFECT_IAM, Alphabet.PERFECT_RIMES]]) > 1:
        raise ValueError('multiple perfect fitting alphabets defeat purpose')
    if Alphabet.PERFECT_GW in alphabet:
        return ALPHABET_PERFECT_GW
    if Alphabet.PERFECT_IAM in alphabet:
        return ALPHABET_PERFECT_IAM
    if Alphabet.PERFECT_RIMES in alphabet:
        return ALPHABET_PERFECT_RIMES
    # universal alphabets
    alph_str = ''
    if Alphabet.ASCII_LOWER in alphabet:
        alph_str += string.ascii_lowercase
    if Alphabet.ASCII_UPPER in alphabet:
        alph_str += string.ascii_uppercase
    if Alphabet.ASCII_DIGITS in alphabet:
        alph_str += string.digits
    if Alphabet.ASCII_PUNCTUATION in alphabet:
        alph_str += string.punctuation
    return alph_str


def occ(k: int, n: int):
    """
    :param k: index of caracter
    :param n: length of word
    :return: occupied interval of k-th character for a word with length n in pct
    """
    return float(k) / float(n), float(k+1) / float(n)


def occ_intersect(intv_0, intv_1):
    """
    :param intv_0: first interval to intersect
    :param intv_1: second interval to intersect
    :return: intersection of intervals
    """
    x0, y0 = intv_0
    x1, y1 = intv_1
    if x0 > y1 or y0 < x1:
        return None
    return max(x0, x1), min(y0, y1)


def occ_abs(intv):
    """
    :param intv: interval
    :return: absolute distance of interval
    """
    return intv[1] - intv[0]


def is_occ(intv_char, intv_reg):
    """
    determines whether an interval is occupied by region based on a 50% overlap threshold

    :param intv_char: interval occupied by the character
    :param intv_reg: interval of region
    :return: boolena indicating occupation
    """
    intv_itsct = occ_intersect(intv_char, intv_reg)
    if intv_itsct is None:
        return False
    abs_intv_intsct = occ_abs(intv_itsct)
    abs_intv_reg = occ_abs(intv_char)
    return abs_intv_intsct / abs_intv_reg >= 0.5


def phoc_levels(word: str, levels=DEFAULT_PHOC_LEVELS):
    """
    calculates the substrings per level of a PHOC for a given word

    :param word: word to generate PHOC for
    :param levels: levels of PHOC
    :return: list containing all substrings of the PHOC for its respective levels
    """
    if levels <= 1:
        return [word]
    # cut_len = float(len(word)) / float(levels)
    # # length of cuts
    # regions = []
    # c_start = 0
    # list of strings
    substrings = []
    regions = [occ(i, levels) for i in range(levels)]
    # checking for overlapping characters for each individual region
    for reg in regions:
        sub_str = ''
        for idx, char in enumerate(word):
            char_occ = occ(idx, len(word))
            if is_occ(char_occ, reg):
                sub_str += char
        substrings.append(deepcopy(sub_str))
    return phoc_levels(word, levels-1) + substrings


def hoc(word: str, alphabet=[Alphabet.ASCII_LOWER, Alphabet.ASCII_DIGITS, Alphabet.ASCII_PUNCTUATION]):
    """
    :param word: request
    :param alphabet: alphabet used for HOC
    :return: HOC for the respective word
    """
    hoc_chars = alphabet_chars(alphabet)
    hoc_arr = np.zeros(len(hoc_chars), dtype=np.uint8)
    for idx, char in enumerate(hoc_chars):
        if char in word:
            hoc_arr[idx] = 1
    return hoc_arr


def phoc(word: str, alphabet=[Alphabet.ASCII_LOWER, Alphabet.ASCII_DIGITS, Alphabet.ASCII_PUNCTUATION],
         levels=DEFAULT_PHOC_LEVELS):
    """
    creates a PHOC encoding from a word

    :param word: string to be encoded into a PHOC
    :param alphabet: alphabet used for PHOC
    :param levels: levels of PHOC
    :return: np.array of PHOC encoding as np.uint8
    """
    substrings = phoc_levels(word, levels=levels)
    hocs = [hoc(sub_str, alphabet=alphabet) for sub_str in substrings]
    return np.concatenate(hocs).astype(np.uint8)


def len_phoc(levels=DEFAULT_PHOC_LEVELS,
             alphabet=[Alphabet.ASCII_LOWER, Alphabet.ASCII_DIGITS, Alphabet.ASCII_PUNCTUATION]):
    """
    clalculates the length of a PHOC vector for a given number of levels

    :param levels: levels of PHOC
    :param alphabet: alphabet used for PHOC
    :return: length of PHOC
    """
    return len(phoc('', alphabet=alphabet, levels=levels))


def char_err(word: str, estimate: str):
    """
    character error for single words, using dynamic programming.
    :math:`O(mn)` , with  :math:`m = |word|, n = |estimate|`

    :param word: gt word
    :param estimate: estimated word
    :return: character error and relative character error in that order
    """
    c_gt = [c for c in word]
    c_est = [c for c in estimate]
    # initialize table
    d = np.zeros((len(c_gt) + 1) * (len(c_est) + 1), dtype=np.uint8)
    d = d.reshape((len(c_gt) + 1, len(c_est) + 1))
    for i in range(len(c_gt) + 1):
        for j in range(len(c_est) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    # compute path
    for i in range(1, len(c_gt) + 1):
        for j in range(1, len(c_est) + 1):
            if c_gt[i - 1] == c_est[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    c_err = d[len(c_gt)][len(c_est)]
    c_err_pct = c_err / len(word)
    return np.array([c_err, c_err_pct])


def word_err(word: str, estimate: str):
    """
    word error for single words

    :param word: gt word
    :param estimate: estimated word
    :return: equality as float
    """
    return 0 if word == estimate else 1
