from torch import nn
import torch

class Conv1d(nn.Module):
    def __init__(self, list_layers):
        super(Conv1d, self).__init__()
        self.conv1d = nn.ModuleList([nn.LazyConv1d(args[0], args[1], 1, padding = 1) for args in list_layers])
        self.activation = nn.LeakyReLU()
    def forward(self, x):
        for layer in self.conv1d:
            x = layer(x)
            x = self.activation(x)
        return x