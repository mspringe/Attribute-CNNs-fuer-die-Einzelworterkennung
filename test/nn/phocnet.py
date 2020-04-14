"""
PHOCNet relevant testing
.. codeauthor:: Maximilian Springenberg <maximilian.springenberg@tu-dortmund.de>
"""
from unittest import TestCase

import numpy as np
import torch
from torch import nn
import torch_testing as tt
# from src.nn.gpp import GPP
from src.nn.phocnet import PHOCNet


class TestPHOCNet(TestCase):

    def setUp(self):
        self.phocnet = PHOCNet(1)
        pass

    def test___init__(self):
        PHOCNet(1)

    def test_forward(self):
        d_img = 1
        batch_size = 1
        phoc_net = PHOCNet(input_channels=d_img, n_out=640)
        img = np.zeros(shape=(batch_size, d_img, 100, 200), dtype=np.float32)
        img[:, :, :, 100:] = 1.
        img = torch.from_numpy(img)
        phoc_net(img)

    def test_pool(self):
        img_arr = np.zeros(shape=(1,1,500,500))
        img_arr[:,:,200:,:200] = 1
        img_tens = torch.from_numpy(img_arr)
        phoc_net = PHOCNet(n_out=1)
        pooled = phoc_net.pool(img_tens)
        compared = torch.nn.functional.max_pool2d(img_tens, kernel_size=phoc_net.kernel_pooling,
                                                  stride=phoc_net.stride_pooling, padding=phoc_net.padding_pooling)
        tt.assert_equal(pooled, compared)

    def test__sanitizeforward(self):
        # set of test cases
        layers = [nn.Linear(1, 1), nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)]
        types_false = [str, int, nn.Module]
        types_true = [nn.Linear, nn.Conv2d]

        x_ins_true = [[1], np.array([1]), torch.tensor([1])]
        x_ins_false = [object(), [1, 2], np.array([1, 2]), torch.tensor([1, 2])]
        # checking layer types
        ref_tens = torch.tensor([1])
        for l, t in zip(layers, types_false):
            self.assertRaises(ValueError, PHOCNet._sanitize_forward, l, ref_tens, 1, t, torch.Tensor, 0)
        for l, t in zip(layers, types_true):
            self.assertEqual(ref_tens, PHOCNet._sanitize_forward(l, ref_tens, 1, t, ch_pos=0))
        # checking inputs
        ref_layer = nn.Linear(1, 1)
        ref_tens = torch.tensor([1]),
        for l, x in zip(layers, x_ins_false):
            self.assertRaises(ValueError, PHOCNet._sanitize_forward, ref_layer, x, 1, nn.Conv2d, torch.Tensor, 0)
        for l, x in zip(layers, x_ins_true):
            tmp_tens = PHOCNet._sanitize_forward(ref_layer, x, 1, nn.Linear, ch_pos=0)
            self.assertIn(tmp_tens, ref_tens)

    def test_convolute(self):
        x_in = torch.from_numpy(np.vstack((np.ones((2, 5, 1, 1),  dtype=np.float32),
                                           np.zeros((1, 5, 1, 1), dtype=np.float32),
                                           np.zeros((2, 5, 1, 1), dtype=np.float32)-1.)))
        # set of test cases
        layer_false = nn.Linear(1, 1)
        layer_true = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=1)

        # checking bad input
        self.assertRaises(Exception, PHOCNet.convolute, layer_false, x_in)
        # checking good input
        PHOCNet.convolute(layer_true, x_in)

    def test_linear_act(self):
        x_in = torch.tensor(np.vstack((np.ones((2, 5),  dtype=np.float32),
                                       np.zeros((1, 5), dtype=np.float32),
                                       np.zeros((2, 5), dtype=np.float32)-1.)))
        # set of test cases
        layer_false = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=1)
        layer_true = nn.Linear(5, 5)

        # checking bad input
        self.assertRaises(Exception, PHOCNet.linear_act, layer_false, x_in)
        # checking good input
        PHOCNet.linear_act(layer_true, x_in)

    def test_linear_sigmoid(self):
        x_in = torch.tensor(np.vstack((np.ones((2, 5),  dtype=np.float32),
                                       np.zeros((1, 5), dtype=np.float32),
                                       np.zeros((2, 5), dtype=np.float32)-1.)))
        # set of test cases
        layer_false = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=1)
        layer_true = nn.Linear(5, 5)

        # checking bad input
        self.assertRaises(Exception, PHOCNet.linear_sigmoid, layer_false, x_in)
        # checking good input
        PHOCNet.linear_act(layer_true, x_in)

    def test_linear_dropout(self):
        phoc_net = PHOCNet(1)
        x_in = torch.tensor(np.vstack((np.ones((2,5),  dtype=np.float32),
                                       np.zeros((1,5), dtype=np.float32),
                                       np.zeros((2,5), dtype=np.float32)-1.)))
        # set of test cases
        layer_false = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=1)
        layer_true = nn.Linear(5,5)

        # checking bad input
        self.assertRaises(Exception, phoc_net.linear_dropout, layer_false, x_in)
        # checking good input
        phoc_net.linear_dropout(layer_true, x_in)
