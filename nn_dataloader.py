#!/usr/bin/env python3

# References:
# [1] https://github.com/kuangliu/pytorch-cifar
# [2] https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# [3] https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

import os
import sys
import torch
import numpy as np
import math
import copy


class NNDataLoader():
    """ dataloader needs to receive a dataset object (see datasets.py) """

    def __init__(self, data_obj, batch_size=64, num_workers=4, **kwargs):
        """ """
        self.data = data_obj
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available()



    def get_loader_in_memory(self, inputs, labels):
        """ """
        dataloader = list(zip(inputs.split(self.batch_size),
                                labels.split(self.batch_size)))

        return dataloader



    @staticmethod
    def shuffle_data_in_memory(inputs, labels):
        """ Shuffle data when tensors are in memory """

        permutation_idx = torch.randperm(inputs.size(0))
        inputs = torch.index_select(inputs, 0, permutation_idx)
        labels = torch.index_select(labels, 0, permutation_idx)

        return inputs, labels



    def get_shuffled_trainloader_in_memory(self):
        """ Get reshufled trainloader when tensors are in memory """

        inputs, labels = self.shuffle_data_in_memory(self.data.train['input'],
                                                    self.data.train['values'])
        trainloader = self.get_loader_in_memory(inputs, labels)

        return trainloader



    def get_train_valid_loader(self):
        """ """
        trainloader = self.get_shuffled_trainloader_in_memory()
        validloader = self.get_loader_in_memory(self.data.valid['input'],
                                                self.data.valid['values'])

        return trainloader, validloader



    def get_test_loader(self):
        """ """
        testloader = self.get_loader_in_memory(self.data.test['input'],
                                            self.data.test['values'])

        return testloader
