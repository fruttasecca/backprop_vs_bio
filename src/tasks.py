#!/usr/bin/python3

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np


class intData(Dataset):

    def __init__(self, device, seq_len, min, max, dataset_size, linear_inc, seed):
        """
        sequences of integer numbers
        :param device: device
        :param seq_len: length of a single sequence
        :param min, max: define range of generated integers
        :param dataset_size: size of the whole dataset
        :param seed: manual seed
        :param linear_inc: defines how often the sequence length will be incremented by 1, starting from
                           seq_len parameter value
        """
        self.device = device
        self.seq_len = seq_len
        self.max = max
        self.min = min
        self.dataset_size = dataset_size
        self.rnd = np.random.RandomState(seed)
        self.linear_inc = linear_inc
        self.cnt = 0

    def __getitem__(self, index):
        if self.linear_inc != None:
            if self.cnt == self.linear_inc:
                self.cnt = 0
                self.seq_len += 1
            self.cnt += 1

        x = self.rnd.randint(self.min, self.max, self.seq_len)
        x1 = torch.tensor(x, device=self.device, dtype=torch.float).unsqueeze(1)
        return x1, x1

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.dataset_size


def intDataLoader(device, batch_size=2, seq_len=2, min=0, max=5, total_batches=8, linear_inc=2, seed=0):
    """
    :param linear_inc : how often in terms of batches the sequence will be incremented: every batch_size * linear_inc
                        sequences the length of generated sequence will be incremented by 1
    :param total_batches : defines the dataset size in terms of batches
    """
    dataset_size = total_batches * batch_size
    if linear_inc != None:
        linear_inc = linear_inc * batch_size
    loader = DataLoader(intData(device, seq_len=seq_len, min=min, max=max, dataset_size=dataset_size,
                                linear_inc=linear_inc, seed=seed), batch_size=batch_size)
    loader.__str__ = "intData"
    parameters_dict = dict()
    parameters_dict["batch_size"] = batch_size
    parameters_dict["seq_len"] = seq_len
    parameters_dict["min"] = min
    parameters_dict["max"] = max
    parameters_dict["dataset_size"] = dataset_size
    parameters_dict["total_batches"] = total_batches
    parameters_dict["linear_inc"] = linear_inc
    parameters_dict["seed"] = seed
    loader.parameters = lambda: parameters_dict
    return loader


class classData(Dataset):

    def __init__(self, device, seq_len, num_classes, dataset_size, linear_inc, seed):
        """
        sequences of one hot encoded classes
        :param device: device
        :param seq_len: length of a single sequence
        :param num_classes: number of classes
        :param dataset_size: size of the whole dataset
        :param seed: manual seed
        :param linear_inc: defines how often the sequence length will be incremented by 1, starting from
                           seq_len parameter value
        """
        self.device = device
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.dataset_size = dataset_size
        self.device = device
        self.rnd = np.random.RandomState(seed)
        self.linear_inc = linear_inc
        self.cnt = 0

        self.datax = []
        self.datay = []

        for _ in range(self.dataset_size):
            x = self.rnd.randint(0, self.num_classes, self.seq_len)
            x_onehot = np.zeros((x.size, self.num_classes))
            x_onehot[np.arange(x.size), x] = 1
            x1 = torch.tensor(x_onehot, device=self.device, dtype=torch.float).unsqueeze(0)
            y = torch.tensor(x, device=self.device, dtype=torch.long).unsqueeze(0)
            self.datax.append(x1)
            self.datay.append(y)

        self.datax = torch.cat(self.datax)
        self.datay = torch.cat(self.datay)

    def __getitem__(self, index):
        return self.datax[index], self.datay[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.dataset_size


def classDataLoader(device, batch_size=2, seq_len=2, num_classes=4, total_batches=8, linear_inc=2, seed=0):
    """
    :param linear_inc : how often in terms of batches the sequence will be incremented: every batch_size * linear_inc
                        sequences the length of generated sequence will be incremented by 1
    :param total_batches : defines the dataset size in terms of batches
    """
    dataset_size = total_batches * batch_size
    if linear_inc != None:
        linear_inc = linear_inc * batch_size
    loader = DataLoader(classData(device, seq_len=seq_len, num_classes=num_classes, dataset_size=dataset_size,
                                  linear_inc=linear_inc, seed=seed), batch_size=batch_size)
    loader.__str__ = "classData"
    parameters_dict = dict()
    parameters_dict["batch_size"] = batch_size
    parameters_dict["seq_len"] = seq_len
    parameters_dict["num_classses"] = num_classes
    parameters_dict["dataset_size"] = dataset_size
    parameters_dict["total_batches"] = total_batches
    parameters_dict["linear_inc"] = linear_inc
    parameters_dict["seed"] = seed
    loader.parameters = lambda: parameters_dict
    return loader


class realValuedData(Dataset):
    """
    Class to generate real valued data.
    """

    def __init__(self, device, dimension, dataset_size, seed):
        """
        :param device: device
        :param dimension: dimension of the output
        :param dataset_size: size of the whole dataset
        :param seed: manual seed
        """
        self.device = device
        self.dimension = dimension
        self.dataset_size = dataset_size
        self.rnd = np.random.RandomState(seed)

        # set mean and covariance for the multivariate distribution used to sample
        self.covariance = self.rnd.uniform(size=self.dimension ** 2).reshape((self.dimension, self.dimension))
        self.covariance = (self.covariance + np.transpose(self.covariance)) / 2

        # needed to mak matrix positive semi definite (just a temporary
        # workaround for the dimensions we are using in the experiments)
        scale = 0.5
        if dimension > 4:
            scale = 1.
        self.covariance += np.identity(self.dimension) * scale

        self.mean = self.rnd.uniform(size=dimension)

        data = []
        for _ in range(self.dataset_size):
            x = self.rnd.multivariate_normal(self.mean, self.covariance, size=1).reshape(self.dimension)
            x1 = torch.tensor(x, device=self.device, dtype=torch.float).view(1, -1)
            data.append(x1)
        self.data = torch.cat(data)

    def __getitem__(self, index):
        return self.data[index], self.data[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.dataset_size


def realValuedDataLoader(device, batch_size=2, dimension=2, total_batches=8, seed=0):
    """
    :param dimension: dimension of the output
    :param total_batches : defines the dataset size in terms of batches
    """
    dataset_size = total_batches * batch_size
    loader = DataLoader(realValuedData(device, dimension, dataset_size=dataset_size, seed=seed), batch_size=batch_size)

    loader.__str__ = "realValuedData"
    parameters_dict = dict()
    parameters_dict["batch_size"] = batch_size
    parameters_dict["dimension"] = dimension
    parameters_dict["dataset_size"] = dataset_size
    parameters_dict["total_batches"] = total_batches
    parameters_dict["seed"] = seed
    loader.parameters = lambda: parameters_dict
    return loader
