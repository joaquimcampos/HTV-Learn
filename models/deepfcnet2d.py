#!/usr/bin/env python3

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class DeepfcNet2D(nn.Module):
    """
    input size :
    N x 2 (point in space)

    Output size of each layer:
    N x 2 -> fc1 -> N x h
          -> fc2 -> N x h
          -> fc3 -> N x h
          -> fc4 -> N x h
          -> fc5 -> N x h
          -> fc6 -> N x h
          -> fc7 -> N x h
          -> fc8 -> N x h
          -> fc9 -> N x h
          -> fc10 -> N x h
          -> fc11 -> N x 1
    """

    def __init__(self, hidden=40, **kwargs):

        super().__init__()
        self.hidden = hidden # number of hidden neurons

        activation_channels = []

        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden, hidden)
        # self.fc6 = nn.Linear(hidden, hidden)
        # self.fc7 = nn.Linear(hidden, hidden)
        # self.fc8 = nn.Linear(hidden, hidden)
        # self.fc9 = nn.Linear(hidden, hidden)
        # self.fc10 = nn.Linear(hidden, hidden)
        self.fc6 = nn.Linear(hidden, 1)

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
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        # x = F.relu(self.fc8(x))
        # x = F.relu(self.fc9(x))
        # x = F.relu(self.fc10(x))
        x = self.fc6(x).squeeze(1)

        return x
