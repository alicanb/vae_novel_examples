from numbers import Number

import numpy as np
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Encoder(nn.Module):
    def __init__(self,
                 im_shape=784,
                 num_hidden=400,
                 dim_z=10,
                 ):

        super(self.__class__, self).__init__()
        if im_shape is None:
            im_shape = [28, 28]
        self.im_shape = im_shape
        num_pixels = np.prod(im_shape)

        if isinstance(num_hidden, Number):
            num_hidden = [num_hidden]

        self.filename = '_'.join(['enc_mlp'] + [str(i) for i in num_hidden])

        h = [Flatten()]
        prev = num_pixels
        if num_hidden:
            for i in num_hidden:
                h.append(nn.Linear(prev, i))
                h.append(nn.ReLU())
                prev = i
        else:
            # num_hidden = [] case
            num_hidden = [num_pixels]
        self.enc_hidden = nn.Sequential(*h)
        self.z_mean = nn.Linear(num_hidden[-1], dim_z)
        self.z_log_std = nn.Linear(num_hidden[-1], dim_z)

    def forward(self, x):
        hidden = self.enc_hidden(x)
        z_mean = self.z_mean(hidden)
        z_log_std = self.z_log_std(hidden)
        return z_mean, z_log_std

    @property
    def num_layers(self):
        # TODO: add other layers
        return sum([1 if isinstance(layer, nn.Linear) else 0 for layer in self.enc_hidden]) + 2


class Decoder(nn.Module):
    def __init__(self,
                 im_shape=None,
                 num_hidden=400,
                 dim_z=10,
                 ):

        super().__init__()
        if im_shape is None:
            im_shape = [28, 28]
        self.im_shape = im_shape
        num_pixels = np.prod(im_shape)
        if isinstance(num_hidden, Number):
            num_hidden = [num_hidden]

        self.filename = '_'.join(['enc_dec'] + [str(i) for i in num_hidden])

        h = []
        prev = dim_z
        if num_hidden:
            for i in num_hidden:
                h.append(nn.Linear(prev, i))
                h.append(nn.ReLU())
                prev = i
        else:
            # num_hidden = [] case
            num_hidden = [dim_z]
        self.dec_hidden = nn.Sequential(*h)
        self.dec_x = nn.Sequential(nn.Linear(num_hidden[-1], num_pixels),
                                   nn.Sigmoid())

    def forward(self, z):
        h = self.dec_hidden(z)
        x_hat = self.dec_x(h)
        return x_hat.reshape(x_hat.shape[0], *self.im_shape)

    @property
    def num_layers(self):
        # TODO: add other layers
        return sum([1 if isinstance(layer, nn.Linear) else 0 for layer in self.dec_hidden]) + 1
