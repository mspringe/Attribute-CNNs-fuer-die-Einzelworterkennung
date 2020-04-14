"""
unit tests for PRM estimator
.. codeauthor:: Maximilian Springenberg <maximilian.springenberg@tu-dortmund.de>
"""
import matplotlib.pyplot as plt
import random
import numpy as np
# unit-test relevant imports
from unittest import TestCase
# own code
from src.estimation import prob
from src.util.phoc_util import phoc


class TestProbEstimator(TestCase):

    def setUp(self):
        # all combinations of the characters "cat"
        self.words = []
        for c1 in 'cat':
            self.words.append(c1)
            for c2 in 'cat':
                self.words.append(c1+c2)
                for c3 in 'cat':
                    self.words.append(c1+c2+c3)
        # estimator
        self.prm = prob.ProbEstimator(self.words)

    def test_estimate(self):
        # estimate and check for results
        query = [phoc('cat')]
        query_words = ['cat']
        self.assertEqual(self.prm.estimate_set(query), query_words)
