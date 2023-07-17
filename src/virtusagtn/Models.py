import torch as _torch
from torch import nn as _nn
from .utils import _weights_init
from torch.nn.utils.weight_norm import weight_norm as _weight_norm
import numpy as _np
import typing as _typing


class Teacher2(_nn.Module):
    ''' Sample Teacher Architecture. Should accept noise and labels to
    output data of shape similar to actual data.
    '''
    def __init__(self,
                 img_size: _typing.List[int],
                 num_classes: int,
                 noise_size: int = 128):
        '''
        Parameters:
            img_size: List of image shape in form : ``[channels, height, width]``

            num_classes: Number of classes in the classification problem.

            noise_size: Number of dimensions of the noise that the Teacher accepts.
        '''
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.noise_size = noise_size
        self.total_dim = noise_size+num_classes

        self.model = _nn.Sequential(
            _weight_norm(_nn.Linear(self.total_dim, self.img_size[1]//4 * self.img_size[2]//4 * self.total_dim)),
            _nn.SELU(),
            _nn.Unflatten(1, (self.total_dim, self.img_size[1]//4, self.img_size[2]//4)),

            _nn.ConvTranspose2d(self.total_dim, 128, 4, 2, padding=1),
            _nn.BatchNorm2d(128),
            _nn.SELU(),

            _nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            _nn.BatchNorm2d(64),
            _nn.SELU(),

            _nn.ConvTranspose2d(64, self.img_size[0], 3, 1, padding=1),
            _nn.Tanh()
            )
        self.model.apply(_weights_init)

    def forward(self, x, noise):
        x = _torch.cat([x, noise], dim=1)
        return self.model(x)

# out_dim = strides * (in_dim - 1) + kernel_size - 2*padding


class Teacher1(_nn.Module):
    ''' Sample Teacher Architecture. Should accept noise and labels to
    output data of shape similar to actual data.
    '''
    def __init__(self,
                 img_size:  _typing.List[int],
                 num_classes: int,
                 noise_size: int = 128):
        '''
        Parameters:
            img_size: List of image shape in form : ``[channels, height, width]``
            num_classes: Number of classes in the classification problem.
            noise_size: Number of dimensions of the noise that the Teacher accepts.
        '''

        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.noise_size = noise_size
        self.total_dim = noise_size+num_classes

        conv1_filters = 64
        fc1_size = 1024
        fc2_filters = 8
        fc2_size = fc2_filters * self.img_size[1] * self.img_size[2]

        self.model = _nn.Sequential(
            _weight_norm(_nn.Linear(self.total_dim, fc1_size)),
            _nn.LeakyReLU(0.1),
            _nn.BatchNorm1d(fc1_size),

            _weight_norm(_nn.Linear(fc1_size, fc2_size)),
            _nn.LeakyReLU(0.1),
            _nn.Unflatten(1, (fc2_filters, self.img_size[1], self.img_size[2])),

            _nn.BatchNorm2d(fc2_filters),

            _weight_norm(_nn.Conv2d(fc2_filters, conv1_filters, 3, 1, padding="same")),
            _nn.LeakyReLU(0.1),
            _nn.BatchNorm2d(conv1_filters),

            _weight_norm(_nn.Conv2d(conv1_filters, self.img_size[0], 3, 1, padding="same")),
            _nn.Tanh()
        )
        self.model.apply(_weights_init)

    def forward(self, x, target):
        x = _torch.cat([x, target], dim=1)
        return self.model(x)


class Learner(_nn.Module):
    ''' Sample Learner Module that provides a random CNN architecture.
    '''
    def __init__(self,
                 img_size:  _typing.List[int],
                 num_classes: int,
                 cnn_filters: _typing.Union[_typing.List[int], None] = None,
                 linear_filters: _typing.Union[_typing.List[int], None] = None):
        ''' Learner model.
        Parameters:
            img_size: tuple of (channels, height, width)
            num_classes: number of classes
            cnn_filters: list of ints representing filters for CNN layers if `None`, random values are used
            linear_filters: list of ints representing for Linear layers, if `None`, random values are used
        Returns:
            model (torch.nn.Module): Learner model
        '''
        super().__init__()
        self.model = _nn.Sequential()
        self.img_size = img_size
        ########################################################################################################################################
        if isinstance(cnn_filters, _np.ndarray) or isinstance(cnn_filters, list):
            cnn_layers = len(cnn_filters)
            cnn_filters = list(cnn_filters)
        else:
            cnn_layers = _np.random.randint(2, 4)
            cnn_filters = list(_np.random.randint(low=64, high=256, size=(cnn_layers, )))
        cnn_filters = [self.img_size[0]]+cnn_filters

        for i in range(cnn_layers):
            this_block = self.cnn_block(cnn_filters[i], cnn_filters[i+1])
            # self.model = nn.Sequential(*self.model.children(), *this_block.children())
            self.model.append(this_block)

        self.model.append(_nn.Flatten())
        ########################################################################################################################################
        initial_filters = cnn_filters[-1] * self.img_size[1]//(2**cnn_layers) * self.img_size[2]//(2**cnn_layers)
        if isinstance(linear_filters, _np.ndarray) or isinstance(linear_filters, list):
            linear_layers = len(linear_filters)
            linear_filters = list(linear_filters)
        else:
            linear_layers = _np.random.randint(1, 3)
            linear_filters = list(_np.random.randint(low=64, high=256, size=(linear_layers, )))

        linear_filters = [initial_filters] + linear_filters + [num_classes]

        for i in range(linear_layers+1):
            if i != linear_layers:
                this_block = self.linear_block(linear_filters[i], linear_filters[i+1])
            else:
                this_block = self.linear_block(linear_filters[i], linear_filters[i+1], final=True)
            # self.model = nn.Sequential(*self.model.children(),*this_block.children())
            self.model.append(this_block)
        self.model.apply(_weights_init)

    def cnn_block(self, initial_filters, final_filters):
        this_block = _nn.Sequential(
                    _nn.Conv2d(initial_filters, final_filters, kernel_size=3, stride=1, padding='same'),
                    _nn.BatchNorm2d(final_filters, track_running_stats=False, momentum=0),
                    _nn.LeakyReLU(0.1),
                    _nn.MaxPool2d(2, 2)
                    )
        return this_block

    def linear_block(self, initial_filters, final_filters, final=False):
        if final is False:
            this_block = _nn.Sequential(
                        _nn.Linear(initial_filters, final_filters),
                        _nn.BatchNorm1d(final_filters, track_running_stats=False, momentum=0),
                        _nn.LeakyReLU(0.1),
                        )
        else:
            this_block = _nn.Sequential(
                        _nn.Linear(initial_filters, final_filters))
        return this_block

    def forward(self, x):
        return self.model(x)
