"""
unit tests for the metric/ distance estimator
.. codeauthor:: Maximilian Springenberg <maximilian.springenberg@tu-dortmund.de>
"""
# unit-test relevant imports
from unittest import TestCase
# own code
from src.estimation import base
from src.util.phoc_util import phoc


class TestDistEstimator(TestCase):

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
        self.est = base.DistEstimator(self.words, 'cosine')

    def test_estimate(self):
        # estimate and check for results
        query = [phoc('cat')]
        query_words = ['cat']
        self.assertEqual(self.est.estimate_set(query), query_words)
