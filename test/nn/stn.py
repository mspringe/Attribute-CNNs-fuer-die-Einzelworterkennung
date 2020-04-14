"""
STN relevant testing
.. codeauthor:: Maximilian Springenberg <maximilian.springenberg@tu-dortmund.de>
"""
from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch_testing as tt
# from src.nn.gpp import GPP
from src.nn.stn import STN


class TestSTN(TestCase):

    def setUp(self):
        self.stn = STN(1)

    def test___init__(self):
        STN(1)

    def test_forward(self):
        batch_size = 1
        d_img = 1
        img = np.zeros(shape=(batch_size, d_img, 100, 200), dtype=np.float32)
        img[:, :, :, 100:] = 1.
        img = torch.tensor(img, dtype=torch.float32)
        transformed = self.stn(img).detach().numpy()[0][0]
        plt.subplot(211)
        plt.imshow(img.numpy()[0][0], cmap='bone')
        plt.subplot(212)
        plt.imshow(transformed, cmap='bone')
        plt.show()
