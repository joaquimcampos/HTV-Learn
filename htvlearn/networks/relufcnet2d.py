#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLUfcNet2D(nn.Module):
    """
    2D Fully-connected network with ReLU activations.

    input size: N x 2  (point in space)

    Network (layer type -> output size):
    fc1    -> N x h
    fc2    -> N x h
    ...
    fcL    -> N x h  (L=num_hidden_layers)
    fclast -> N x 1
    """
    def __init__(self,
                 num_hidden_layers=5,
                 num_hidden_neurons=50,
                 **kwargs):
        """
        Args:
            num_hidden_layers (int):
                number of hidden layers
            num_hidden_neurons (int):
                number of hidden neurons
        """

        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_neurons = num_hidden_neurons

        self.fc1 = nn.Linear(2, num_hidden_neurons)
        self.fchidden = nn.ModuleList(
            [nn.Linear(num_hidden_neurons, num_hidden_neurons)
             for i in range(num_hidden_layers - 1)]
        )
        self.fclast = nn.Linear(num_hidden_neurons, 1)

        self.num_params = self.get_num_params()

    def get_num_params(self):
        """ """
        num_params = 0
        for param in self.parameters():
            num_params += torch.numel(param)

        return num_params

    def forward(self, x):
        """ """
        x = F.relu(self.fc1(x))

        for fclayer in self.fchidden:
            x = F.relu(fclayer(x))

        x = self.fclast(x).squeeze(1)

        return x
