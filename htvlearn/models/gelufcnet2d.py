#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELUfcNet2D(nn.Module):
    """
    input size :
    N x 2 (point in space)

    Output size of each layer:
    N x 2 -> fc1 -> N x h
          -> fchidden -> N x N (x num_hidden_layers)
          -> fclast -> N x 1
    """
    def __init__(self,
                 num_hidden_layers=5,
                 num_hidden_neurons=50,
                 **kwargs):

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
        x = F.gelu(self.fc1(x))

        for fclayer in self.fchidden:
            x = F.gelu(fclayer(x))

        x = self.fclast(x).squeeze(1)

        return x
