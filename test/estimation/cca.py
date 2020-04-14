"""
unit tests for RCCA estimator
.. codeauthor:: Maximilian Springenberg <maximilian.springenberg@tu-dortmund.de>
"""
import matplotlib.pyplot as plt
import random
import numpy as np
# unit-test relevant imports
from unittest import TestCase
# own code
from src.estimation import cca
from src.pyrcca.rcca import CCA as RCCA
from src.util.phoc_util import phoc


class TestPYRCCA(TestCase):

    def setUp(self):
        # CCA should learn: 1st dim = 1st dim, discard 2nd dim, 3rd dim = 0.5 2nd dim
        self.attr_3d_train = np.array([[1, 1, 2],
                                       [2, 99, 4],
                                       [7, 4, 6],
                                       [5, 0, 8]])
        self.space_2d_train = np.array([[1, 1],
                                        [2, 2],
                                        [7, 3],
                                        [5, 4]])
        # vector space
        self.space_2d = np.array([[1, 4],
                                  [1, 11],
                                  [2, 5],
                                  [1, 1]]).astype(float)
        # 'estimated' data
        self.attr_3d = np.array([[1, 3, 8],
                                 [1, 5, 22],
                                 [2, 20, 10],
                                 [1, 4, 2]]).astype(float)
        # adding noise
        noise = np.array([[random.uniform(-0.2, 0.2) for j in range(3)] for i in range(len(self.attr_3d))])
        # noise = np.array([[0.01 for j in range(3)] for i in range(len(self.attr_3d))])
        self.attr_3d += noise

    def test_cca(self):
        # Initialize number of samples
        nSamples = 1000

        # Define two latent variables (number of samples x 1)
        latvar1 = np.random.randn(nSamples, )
        latvar2 = np.random.randn(nSamples, )

        # Define independent components for each dataset (number of observations x dataset dimensions)
        indep1 = np.random.randn(nSamples, 4)
        indep2 = np.random.randn(nSamples, 5)

        # Create two datasets, with each dimension composed as a sum of 75% one of the latent variables and 25% independent component
        data1 = 0.25 * indep1 + 0.75 * np.vstack((latvar1, latvar2, latvar1, latvar2)).T
        data2 = 0.25 * indep2 + 0.75 * np.vstack((latvar1, latvar2, latvar1, latvar2, latvar1)).T

        # Split each dataset into two halves: training set and test set
        train1 = data1[:int(nSamples / 2)]
        train2 = data2[:int(nSamples / 2)]
        test1 = data1[int(nSamples / 2):]
        test2 = data2[int(nSamples / 2):]

        # Create a cca object as an instantiation of the CCA object class.
        cca = RCCA(kernelcca=False, reg=1e-6, numCC=2)

        # Use the train() method to find a CCA mapping between the two training sets.
        cca.train([train1, train2])

        test1_trans, test2_trans = [data.dot(wght) for data, wght in zip([test1, test2], cca.ws)]

        # Use the validate() method to test how well the CCA mapping generalizes to the test data.
        # For each dimension in the test data, correlations between predicted and actual data are computed.
        testcorrs = cca.validate([test1, test2])


class TestCCAEstimator(TestCase):

    def setUp(self):
        self.words = ['cat', 'dog', 'fox']
        self.phocs = [phoc(w) for w in self.words]
        self.rcca = cca.RCCAEstimator(self.words)

    def test_estimate(self):
        # train with one hot encodings of different dimensionality to PHOC
        codes = [[0, 0, 1],
                 [0, 1, 0],
                 [1, 0, 0]]
        self.rcca.fit(codes, self.phocs)
        # estimate and check for results
        query = [[0, 1, 0],
                 [0, 0, 1],
                 [1, 0, 0]]
        query_words = ['dog', 'cat', 'fox']
        self.assertEqual(self.rcca.estimate_set(query), query_words)
