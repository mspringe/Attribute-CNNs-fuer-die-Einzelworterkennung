"""
STN implementation for the PHOCNet.
This approach was not followed through and discussed in my thesis.
Hence good results, using this STN are not guaranteed.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
from torch import nn
import torch.nn.functional as F
from src.nn.pp import GPP, PPTypes, PPTypePooling


class STN(nn.Module):
    """
    A simple STN implementation, that can be used as an initial layer in :class:`src.nn.phocnet.STNPHOCNet`.

    For more information on STNs have a look at Max Jaderbergs
    `paper <https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf>`__.
    """

    def __init__(self, input_channels=1):
        super().__init__()
        # convolutional layers
        kernel_size_conv = 3
        padding_conv = 1
        stride_conv = 1
        self.loc_c1 = nn.Conv2d(in_channels=input_channels, out_channels=16,
                                kernel_size=kernel_size_conv, padding=padding_conv, stride=stride_conv)
        self.loc_c2 = nn.Conv2d(in_channels=self.loc_c1.out_channels, out_channels=32,
                                kernel_size=kernel_size_conv, padding=padding_conv, stride=stride_conv)
        self.loc_c3 = nn.Conv2d(in_channels=self.loc_c2.out_channels, out_channels=32,
                                kernel_size=kernel_size_conv, padding=padding_conv, stride=stride_conv)
        # set-up pooling layers
        self.padding_pooling = 0
        self.kernel_pooling = 2
        self.stride_pooling = 2
        # spatial pooling layer max of 1 + 4 + 16 + 64 + 256 = 341 bins per feature map => 341 * 32 = 10912 out_vector
        self.loc_spp = GPP(gpp_type=PPTypes.T_SPP, levels=5,
                           pool_type=PPTypePooling.MAX_POOL, n_f_maps=self.loc_c3.out_channels)
        # regression
        self.loc_lin1 = nn.Linear(self.loc_spp.pooling_output_size, 1024)
        self.loc_out = nn.Linear(self.loc_lin1.out_features, 3*2)

    def forward(self, U):
        """
        Forward pass of this STN, i.e. transformation of the input image/ map U.

        :param U: feature map/ image U
        :return: transformed image
        """
        theta = self.f_loc(U)
        theta = theta.view(-1, 2, 3)
        sampling_grid = self.T_theta(theta, U.size())
        V = self.sampler(feature_map=U, sampling_grid=sampling_grid)
        return V

    def f_loc(self, U):
        """
        The localisation network

        :param U: feature map/ image U
        :return: parameters :math:`\\Theta` for the grid generator
        """
        # convolution
        theta = F.relu(self.loc_c1(U))
        theta = self.pool(theta)
        theta = F.relu(self.loc_c2(theta))
        theta = self.pool(theta)
        theta = F.relu(self.loc_c3(theta))
        # pyramidal pooling
        theta = self.loc_spp(theta)
        # regression values
        theta = F.relu(self.loc_lin1(theta))
        theta = F.relu(self.loc_out(theta))
        return theta

    def pool(self, x_in):
        try:
            # pooling
            x_out = F.max_pool2d(x_in, kernel_size=self.kernel_pooling, stride=self.stride_pooling,
                                 padding=self.padding_pooling)
        except RuntimeError as rte:
            return x_in
        return x_out

    def T_theta(self, theta, size):
        """
        The grid generator, applied to the regular spatial grid

        :param theta: parameters of the grid generator (usually provided by the localisation network)
        :param size: size of the input feature-map/ image
        :return: generated sampling grid
        """
        sampling_grid = F.affine_grid(theta, size)
        return sampling_grid

    def sampler(self, feature_map, sampling_grid):
        """
        The sampler

        :param feature_map: input feature-map/ image
        :param sampling_grid: sampling grid, used for warping
        :return: warped feature-map/ image
        """
        V = F.grid_sample(feature_map, sampling_grid)
        return V

    def setup(self):
        return {'c_in': self.loc_c1.in_channels}
