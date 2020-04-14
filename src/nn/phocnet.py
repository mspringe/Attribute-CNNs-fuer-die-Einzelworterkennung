"""
A PHOCNet implementation.

I took some inspiration from a pre-existing implementation of the
`PHOCNet <https://github.com/georgeretsi/pytorch-phocnet>`__.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
# system libraries
import warnings
# torch relevant imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# own libraries
from src.nn.pp import GPP, PPTypes, PPTypePooling
from src.nn.stn import STN


class PHOCNet(nn.Module):
    """
    Implementation of the PHOCNet architecture proposed by Sebastian Sudhold in his paper
    """

    def __init__(self, n_out, input_channels=1,
                 pp_type=PPTypes.T_TPP, pooling_levels=5, pool_type=PPTypePooling.MAX_POOL):
        """
        :param n_out: number of output channels
        :param input_channels: number of input channels
        :param gpp_type: type of input channels (see :cls:`GPPTypes`)
        :param pooling_levels: levels of the gpp
        :param pool_type: pooling of the gpp (see :cls:`GPPPooling`)
        """
        super(PHOCNet, self).__init__()
        # set-up convolution layers Layers
        kernel_size_conv = 3
        padding_conv = 1
        stride_conv = 1
        # set-up pooling layers
        self.padding_pooling = 0
        self.kernel_pooling = 2
        self.stride_pooling = 2
        # phase 1: conv. 3 x 3 pooling layers + ReLU
        self.conv1_1 = nn.Conv2d(in_channels=input_channels, out_channels=64,
                                 kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv)
        self.conv1_2 = nn.Conv2d(in_channels=self.conv1_1.out_channels,  out_channels=64,
                                 kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv)
        # phase 2: conv. after 1st max-pooling: 3 x 3 pooling layers + ReLU
        self.conv2_1 = nn.Conv2d(in_channels=self.conv1_2.out_channels,  out_channels=128,
                                 kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv)
        self.conv2_2 = nn.Conv2d(in_channels=self.conv2_1.out_channels, out_channels=128,
                                 kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv)
        # phase 3: conv. after 2nd max-pooling: 3 x 3 pooling layers + ReLU
        self.conv3_1 = nn.Conv2d(in_channels=self.conv2_2.out_channels, out_channels=256,
                                 kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv)
        self.conv3_2 = nn.Conv2d(in_channels=self.conv3_1.out_channels, out_channels=256,
                                 kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv)
        self.conv3_3 = nn.Conv2d(in_channels=self.conv3_2.out_channels, out_channels=256,
                                 kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv)
        self.conv3_4 = nn.Conv2d(in_channels=self.conv3_3.out_channels, out_channels=256,
                                 kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv)
        self.conv3_5 = nn.Conv2d(in_channels=self.conv3_4.out_channels, out_channels=256,
                                 kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv)
        self.conv3_6 = nn.Conv2d(in_channels=self.conv3_5.out_channels, out_channels=512,
                                 kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv)
        # phase 4: conv. upscaled channels
        self.conv4_1 = nn.Conv2d(in_channels=self.conv3_6.out_channels, out_channels=512,
                                 kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv)
        self.conv4_2 = nn.Conv2d(in_channels=self.conv4_1.out_channels, out_channels=512,
                                 kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv)
        self.conv4_3 = nn.Conv2d(in_channels=self.conv4_2.out_channels, out_channels=512,
                                 kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv)
        # creating the (spatial for gpp_type='spp') pyramid pooling layer
        self.pooling_layer_fn = GPP(gpp_type=pp_type, levels=pooling_levels, pool_type=pool_type)
        pooling_output_size = self.pooling_layer_fn.pooling_output_size
        # phase 5: linear, fully connected + ReLU(with dropouts)
        self.lin5_1 = nn.Linear(pooling_output_size, 4096)
        self.lin5_2 = nn.Linear(4096, 4096)
        # phase 6: linear, fullly connected + linear activation
        self.lin6_1 = nn.Linear(4096, n_out)
        # phase 7: output layer, fully connected + sigmoid
        self.out = nn.Linear(n_out, n_out)

    def neural_codes(self, x, pos=0):
        """
        Calculates "the neural-codes", in other words the output one layer before the prediction layer.
        Those can be used to create a subspace between the predicted attributes and the neural codes themselfes, in
        order to predict words in that subspace based on the much larger neural codes with more encoded information.
        (e.g. using CCA)

        :param x: input vector of image-data (normalized in [0,1])
        :return: tensor of neural codes
        """
        if not -4 <= pos <= 0:
            raise ValueError('got pos={}, '.format(pos) +
                             'but neural codes are only available for up to 4 layers prior => pos in [-4,0]')
        # phase 1: conv. 3 x 3 pooling layers + ReLU
        y = self.convolute(self.conv1_1, x)
        y = self.convolute(self.conv1_2, y)
        # pooling
        y = self.pool(y)
        # phase 2: conv. after 1st max-pooling: 3 x 3 pooling layers + ReLU
        y = self.convolute(self.conv2_1, y)
        y = self.convolute(self.conv2_2, y)
        # pooling
        y = self.pool(y)
        # phase 3: conv. after 2nd max-pooling: 3 x 3 pooling layers + ReLU
        y = self.convolute(self.conv3_1, y)
        y = self.convolute(self.conv3_2, y)
        y = self.convolute(self.conv3_3, y)
        y = self.convolute(self.conv3_4, y)
        y = self.convolute(self.conv3_5, y)
        y = self.convolute(self.conv3_6, y)
        # phase 4: conv. upscaled channels
        y = self.convolute(self.conv4_1, y)
        y = self.convolute(self.conv4_2, y)
        y = self.convolute(self.conv4_3, y)
        # spatial pooling
        y = self.pooling_layer_fn.forward(y)
        if pos == -4:
            return y
        # phase 5: linear, fully connected + ReLU(with dropouts)
        y = self.linear_dropout(self.lin5_1, y)
        if pos == -3:
            return y
        y = self.linear_dropout(self.lin5_2, y)
        if pos == -2:
            return y
        # phase 6: linear, fullly connected + linear activation
        y = self.linear_act(self.lin6_1, y)
        if pos == -1:
            return y
        # phase 7: output layer, fully connected + sigmoid
        y = self.linear_sigmoid(self.out, y)
        return y

    def forward(self, x):
        """
        performs a forward pass of this network (overrides :func:`nn.Module.forward`)

        :param x: input-vector of image-data (notmalized in [0,1])
        :return: tensor of PHOC-encoding
        """
        # neural codes of this network, (all phases 1-7)
        y = self.neural_codes(x, pos=0)
        return y

    def init_weights(self):
        """
        weights initialization, (overrides :func:`nn.Module`)
        """
        self.apply(PHOCNet._init_weights_he)

    @staticmethod
    def _init_weights_he(m: nn.Module):
        """
        PHOCNet weight initialization to be applied to a Module

        :param m: Module to initialize wieghts for
        """
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
        if isinstance(m, nn.Linear):
            n = m.out_features
            m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
            nn.init.constant_(m.bias.data, 0)

    def pool(self, x_in):
        """
        performs 2d pooling

        :param x_in: input vector
        :return: pooling results
        """
        # sanity checks (preferably working with Tensors)
        if not isinstance(x_in, torch.Tensor):
            warnings.warn('WARNING: expected Tensor, got {}'.format(type(x_in)), Warning, stacklevel=2)
            x_in = torch.tensor(x_in)
        try:
            # pooling
            x_out = F.max_pool2d(x_in, kernel_size=self.kernel_pooling, stride=self.stride_pooling,
                                 padding=self.padding_pooling)
        except RuntimeError as _:
            # input to small for pooling??? -> maybe a single point '.'?
            return x_in
        return x_out

    def linear_dropout(self, layer: nn.Linear, x_in: torch.Tensor):
        """
        performs a linear forward propagation with relu activation and a dropout

        :param layer: layer to propagate forward from
        :param x_in: input vector
        :return: activations
        """
        # sanity checks (Linear dropouts, preferably working with Tensors, correct input size)
        x_in = PHOCNet._sanitize_forward(layer=layer, x_in=x_in, n_channels=layer.in_features, t_layer=nn.Linear)
        # relu activation and dropouts
        x_out = F.relu(layer(x_in))
        x_out = F.dropout(x_out, p=0.5, training=self.training)
        return x_out

    def display_forward(self, x):
        """
        This method displays/ visualizes the outputs of all layers

        :param x:  input tensor
        :return: estimated PHOC
        """
        import matplotlib.pyplot as plt
        import numpy as np
        def display_img(x, count):
            f_maps = x.detach().numpy()[0]
            img = np.max(f_maps, axis=0)
            #plt.subplot(3,8,count)
            plt.imshow(img, cmap='jet')
            plt.show()
        def display_bar(x, count):
            cmap = plt.get_cmap('jet')
            vals = x.detach().numpy()[0]
            colors = cmap(vals)
            #plt.subplot(3,5,10+count)
            plt.bar(np.arange(len(vals)), vals, color=colors, width=1.)
            plt.show()

        c = 1
        display_img(x, c)
        c += 1

        # phase 1: conv. 3 x 3 pooling layers + ReLU
        y = self.convolute(self.conv1_1, x)
        display_img(y, c)
        c += 1
        y = self.convolute(self.conv1_2, y)
        display_img(y, c)
        c += 1

        # pooling
        y = self.pool(y)
        display_img(y, c)
        c += 1
        # phase 2: conv. after 1st max-pooling: 3 x 3 pooling layers + ReLU
        y = self.convolute(self.conv2_1, y)
        display_img(y, c)
        c += 1
        y = self.convolute(self.conv2_2, y)
        display_img(y, c)
        c += 1

        # pooling
        y = self.pool(y)
        display_img(y, c)
        c += 1
        # phase 3: conv. after 2nd max-pooling: 3 x 3 pooling layers + ReLU
        y = self.convolute(self.conv3_1, y)
        display_img(y, c)
        c += 1
        y = self.convolute(self.conv3_2, y)
        display_img(y, c)
        c += 1
        y = self.convolute(self.conv3_3, y)
        display_img(y, c)
        c += 1
        y = self.convolute(self.conv3_4, y)
        display_img(y, c)
        c += 1
        y = self.convolute(self.conv3_5, y)
        display_img(y, c)
        c += 1
        y = self.convolute(self.conv3_6, y)
        display_img(y, c)
        c += 1

        # phase 4: conv. upscaled channels
        y = self.convolute(self.conv4_1, y)
        display_img(y, c)
        c += 1
        y = self.convolute(self.conv4_2, y)
        display_img(y, c)
        c += 1
        y = self.convolute(self.conv4_3, y)
        display_img(y, c)
        c += 1

        # spatial pooling
        c = 1
        y = self.pooling_layer_fn.forward(y)
        display_bar(y, c)
        c += 1

        # phase 5: linear, fully connected + ReLU(with dropouts)
        y = self.linear_dropout(self.lin5_1, y)
        display_bar(y, c)
        c += 1

        y = self.linear_dropout(self.lin5_2, y)
        display_bar(y, c)
        c += 1

        # phase 6: linear, fullly connected + linear activation
        y = self.linear_act(self.lin6_1, y)
        display_bar(y, c)
        c += 1

        # phase 7: output layer, fully connected + sigmoid
        y = self.linear_sigmoid(self.out, y)
        display_bar(y, c)
        c += 1
        # plt.show()

        return y

    @staticmethod
    def linear_sigmoid(layer: nn.Linear, x_in: torch.Tensor):
        """
        linear forward pass with sigmoid activation, no dropouts

        :param layer: linear layer
        :param x_in: input tensor
        """
        # sanity checks (Linear, preferably working with Tensors, correct input size)
        x_in = PHOCNet._sanitize_forward(layer=layer, x_in=x_in, n_channels=layer.in_features, t_layer=nn.Linear)
        # relu activation and dropouts
        x_out = torch.sigmoid(layer(x_in))
        return x_out

    @staticmethod
    def linear_act(layer: nn.Linear, x_in: torch.Tensor):
        """
        simple linear forward pass

        :param layer: linear layer
        :param x_in: input tensor
        """
        # sanity checks (Linear dropouts, preferably working with Tensors, correct input size)
        x_in = PHOCNet._sanitize_forward(layer=layer, x_in=x_in, n_channels=layer.in_features, t_layer=nn.Linear)
        # linear activation
        x_out = layer(x_in)
        return x_out

    @staticmethod
    def convolute(layer: nn.Conv2d, x_in: torch.Tensor):
        """
        perfoms a 2d convolution with relu activation

        :param layer: layer for convolution
        :param x_in: input tensor
        :return: convoluted and activated output
        """
        # sanity checks (only nn.Conv2d layers make sense, preferably working with Tensors, correct input size)
        x_in = PHOCNet._sanitize_forward(layer=layer, x_in=x_in, n_channels=layer.in_channels, t_layer=nn.Conv2d)
        # convolution and activation
        x_out = F.relu(layer(x_in))
        return x_out

    @staticmethod
    def _sanitize_forward(layer, x_in, n_channels, t_layer, t_in=torch.Tensor, ch_pos=1):
        """
        Performs sanity checks on layer and input data. Corrects input data if necessary.

        :param layer: layer to be checked
        :param x_in: input data to be checked
        :param t_layer: expected type of layer
        :param t_in: expected type of input data
        :return: correct input
        """
        if not isinstance(layer, t_layer):
            raise ValueError('expected type {}, got type {}'.format(t_layer, type(layer)))
        if not isinstance(x_in, t_in):
            warnings.warn('WARNING: expected Tensor, got {}'.format(type(x_in)), Warning, stacklevel=3)
            x_in = torch.tensor(x_in)
        if not n_channels == x_in.size(dim=ch_pos):
            raise ValueError('expected input size {}, got {}'.format(n_channels, x_in.size(dim=ch_pos)))
        return x_in

    def setup(self):
        pp_type = str(self.pooling_layer_fn.gpp_type)
        pp_pooling_type = str(self.pooling_layer_fn.pool_type)
        return {'pp_type': pp_type, 'pp_poolong_type': pp_pooling_type,
                'pp_pooling_levels': self.pooling_layer_fn.levels, 'n_out': self.out.out_features,
                'c_in': self.conv1_1.in_channels}


class STNPHOCNet(PHOCNet):
    """
    PHOCNet with initial STN layer.


    .. note::

        The STN does not have to be a the STN implemented in :class:`src.nn.stn.STN`.
        In fact it can be any model, leaving you the option to use custom STN models.
    """

    def __init__(self, n_out, stn=None, input_channels=1,
                 pp_type=PPTypes.T_TPP, pooling_levels=5, pool_type=PPTypePooling.MAX_POOL):
        super().__init__(n_out=n_out, input_channels=input_channels, pp_type=pp_type, pooling_levels=pooling_levels,
                         pool_type=pool_type)
        # sanity checks for STN layer
        valid_stn = False
        if stn is not None:
            if isinstance(stn, STN):
                valid_stn = True
        if not valid_stn:
            self.stn = STN(input_channels=input_channels)
        else:
            self.stn = stn

    def neural_codes(self, x, pos=0):
        """
        Performs regular fowardpass with the STN up front.

        :param x: input image
        :param pos: layer-position (offset from the output layer) to extract neural codes from
        :return: neural code
        """
        x_trans = self.stn(x)
        y = super().neural_codes(x_trans, pos=pos)
        return y
