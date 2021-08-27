#!/usr/bin/env python3

# References:
# [1] https://github.com/kuangliu/pytorch-cifar
# [2] https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# [3] https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

import torch


class NNDataLoader():
    def __init__(self,
                 data_obj,
                 batch_size=64,
                 num_workers=4,
                 **kwargs):
        """
        Args:
            data_obj (Data):
                instance of Data class (data.py).
            batch_size (int):
                batch size for neural network.
            num_workers (int):
                number of subprocesses to use for data loading.
        """
        self.data = data_obj
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available()

    def get_loader_in_memory(self, inputs, labels, batch_size=None):
        """
        Split the data in batches.

        Args:
            inputs (torch.Tensor)
            labels (torch.Tensor)
            batch_size (int):
                loader batch size. If None, self.batch_size is used.

        Returns:
            dataloader (iter):
                iterable through batches inputs-label pairs.
        """
        minibatch = self.batch_size if batch_size is None else batch_size
        dataloader = list(zip(inputs.split(minibatch),
                              labels.split(minibatch)))

        return dataloader

    @staticmethod
    def shuffle_data_in_memory(inputs, labels):
        """
        Shuffle data when tensors are in memory.

        Args:
            inputs (torch.Tensor)
            labels (torch.Tensor)

        Returns:
            inputs (torch.Tensor):
                shuffled inputs.
            labels (torch.Tensor):
                labels corresponding to shuffled inputs.
        """

        permutation_idx = \
            torch.randperm(inputs.size(0)).to(device=inputs.device)
        inputs = torch.index_select(inputs, 0, permutation_idx)
        labels = torch.index_select(labels, 0, permutation_idx)

        return inputs, labels

    def get_shuffled_trainloader_in_memory(self):
        """
        Get reshufled trainloader when tensors are in memory.

        Shuffles the data and splits it in batches.

        Returns:
            trainloader (iter):
                training set iterator of input-label batch pairs.
        """
        train_inputs, train_labels = \
            self.shuffle_data_in_memory(self.data.train['input'],
                                        self.data.train['values'])
        trainloader = self.get_loader_in_memory(train_inputs, train_labels)

        return trainloader

    def get_train_valid_loader(self):
        """
        Get the training and validation loaders for batch training.

        Returns:
            trainloader (iter):
                training set iterator of input-label batch pairs.
            validloader (iter):
                validation set iterator  of input-label batch pairs.
        """
        trainloader = self.get_shuffled_trainloader_in_memory()
        validloader = self.get_loader_in_memory(self.data.valid['input'],
                                                self.data.valid['values'])

        return trainloader, validloader

    def get_test_loader(self):
        """
        Get the test loader for batch testing.

        Returns:
            testloader (iter):
                test set iterator of input-label batch pairs.
        """
        testloader = self.get_loader_in_memory(self.data.test['input'],
                                               self.data.test['values'])

        return testloader
