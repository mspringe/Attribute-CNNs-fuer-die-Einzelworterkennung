"""
Implmentation of different pyramidal pooling layers

    * SPP: spatial pyramidal pooling (vertical and horizontal bins)
    * TPP: temporal pyramidal pooling (horizontal bins only)


I took some inspiration from a pre-existing implementation of the
`PHOCNet <https://github.com/georgeretsi/pytorch-phocnet>`__.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
from copy import deepcopy
from enum import Enum
import torch
from torch import nn
import torch.nn.functional as F


class PPTypes(Enum):
    """unique values for pyramidal pooling-types/versions"""
    T_SPP = 2
    T_TPP = 3


class PPTypePooling(Enum):
    """unique values for pyramidal pooling-pooling procedure"""
    MAX_POOL = 4
    AVG_POOL = 5


class GPP(nn.Module):
    """Generic class for SPP and TPP layers"""

    def __init__(self, gpp_type=PPTypes.T_TPP, levels=3, pool_type=PPTypePooling.MAX_POOL, n_f_maps=512):
        """
        :param gpp_type: pyramidal pooling version/ type
        :param levels: levels of pyramid/ complete binary tree
        :param pool_type: pooling procedure
        """
        super().__init__()
        # setting globals
        self.__gpp_type = None
        self.__pool_type = None
        self.pooling_output_size = None
        self.levels = levels
        self.n_f_maps = n_f_maps
        # setting properties
        self.pool_type = pool_type
        self.gpp_type = gpp_type

    @property
    def gpp_type(self):
        return deepcopy(self.__gpp_type)

    @gpp_type.setter
    def gpp_type(self, gpp_type):
        """
        :param gpp_type: pyramidal pooling version/ type
        """
        # sanity
        if gpp_type not in PPTypes:
            raise ValueError('Unknown gpp_type. Must be in {}'.format(PPTypes))
        if gpp_type == PPTypes.T_SPP:
            self.pooling_output_size = sum([4 ** level for level in range(self.levels)]) * self.n_f_maps
        elif gpp_type == PPTypes.T_TPP:
            self.pooling_output_size = (2 ** self.levels - 1) * self.n_f_maps
            #self.pooling_output_size = sum([level + 1 for level in range(self.levels)]) * self.n_f_maps
        self.__gpp_type = gpp_type

    @property
    def pool_type(self):
        return deepcopy(self.__pool_type)

    @pool_type.setter
    def pool_type(self, pool_type):
        """
        :param pool_type: pooling procedure
        """
        # sanity
        if pool_type not in PPTypePooling:
            raise ValueError('Unknown pool_type. Must be in{}'.format(PPTypePooling))
        self.__pool_type = pool_type

    def forward(self, input_x):
        """
        :param input_x: input of filters
        :return: Vector of concatenated, pooled bins
        """
        # spatial pyramid forward pass
        if self.gpp_type == PPTypes.T_SPP:
            return self._spatial_pyramid_pooling(input_x, self.levels)
        # temporal pyramid forward pass
        elif self.gpp_type == PPTypes.T_TPP:
            #return self._temporal_pyramid_pooling(input_x, self.levels)
            return self._temporal_pyramid_pooling_BT(input_x, self.levels)
        else:
            raise AttributeError('global gpp_type not in {}'.format(PPTypes))

    def _pyramid_pooling(self, input_x, output_sizes):
        """
        :param input_x: input of filters
        :param output_sizes: tuple containing numbers of vertical and horizontal bins, in that order
        :return: Vector of concatenated, respectively pooled bins
        """
        pyramid_level_tensors = []
        for tsize in output_sizes:
            if self.pool_type == PPTypePooling.MAX_POOL:
                pyramid_level_tensor = F.adaptive_max_pool2d(input_x, tsize)
            if self.pool_type == PPTypePooling.AVG_POOL:
                pyramid_level_tensor = F.adaptive_avg_pool2d(input_x, tsize)
            pyramid_level_tensor = pyramid_level_tensor.view(input_x.size(0), -1)
            pyramid_level_tensors.append(pyramid_level_tensor)
        return torch.cat(pyramid_level_tensors, dim=1)

    def _spatial_pyramid_pooling(self, input_x, levels):
        """
        :param input_x: input of filters
        :param levels: levels of pyramid/ complete binary tree
        :return: spatially pooled bins
        """
        output_sizes = [(int(2 ** level), int(2 ** level)) for level in range(levels)]
        return self._pyramid_pooling(input_x, output_sizes)

    def _temporal_pyramid_pooling_BT(self, input_x, levels):
        """
        temporal pyramidal pooling with bin-tree structure (exponential grow of levels with base 2)

        :param input_x: input of filters
        :param levels: levels of pyramid/ complete binary tree
        :return: temporally pooled bins
        """
        output_sizes = [(1, int(2 ** level)) for level in range(levels)]
        return self._pyramid_pooling(input_x, output_sizes)

    def _temporal_pyramid_pooling(self, input_x, levels):
        """
        classic temporal pyramidal pooling

        :param input_x: input of filters
        :param levels: levels of pyramid/ complete binary tree
        :return: temporally pooled bins
        """
        output_sizes = [(1, level+1) for level in range(levels)]
        return self._pyramid_pooling(input_x, output_sizes)
