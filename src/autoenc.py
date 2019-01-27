import torch.nn as nn
from collections import OrderedDict

"""
Simple auto encoder architecture used in this project, based on fully connected layers.
"""


class AutoEncoder(nn.Module):
    def __init__(self, device, input_dim, nonlinearity=False):
        """
        :param device: Device to which to map tensors (GPU or CPU).
        :param input_dim: Input size.
        """
        super(AutoEncoder, self).__init__()

        self.device = device
        self.input_dim = input_dim

        layers = OrderedDict()
        layers["0"] = nn.Linear(input_dim, input_dim // 2)
        layers["1"] = nn.Linear(input_dim // 2, input_dim // 4)
        layers["2"] = nn.Linear(input_dim // 4, input_dim // 2)
        layers["3"] = nn.Linear(input_dim // 2, input_dim)

        if nonlinearity:
            layers["0"] = nn.Linear(input_dim, input_dim // 2)
            layers["1"] = nn.ReLU()
            layers["2"] = nn.Linear(input_dim // 2, input_dim // 4)
            layers["3"] = nn.ReLU()
            layers["4"] = nn.Linear(input_dim // 4, input_dim // 2)
            layers["5"] = nn.Linear(input_dim // 2, input_dim)

        self.layers = nn.Sequential(layers)

    def forward(self, batch):
        """
        Forward pass of the autoencoder.
        :param batch: Input batch (batch length, dim of input).
        :return: Output of the same size of the input,
        """
        res = self.layers(batch)
        return res

    def total_parameters(self):
        """
        Return the total number of trainable parameters.
        :return:
        """
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def custom_flatten_parameters(self):
        pass
