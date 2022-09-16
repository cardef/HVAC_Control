from torch import nn
import torch

class Conv1d(nn.Module):
    def __init__(self, list_layers):
        super(Conv1d, self).__init__()
        self.conv1d = nn.ModuleList([nn.LazyConv1d(args[1], args[2], args[3], padding = args[4]) for args in list_layers])
        self.activation = nn.ReLU()
    def forward(self, x):
        for layer in self.conv1d:
            x = layer(x)
            x = self.activation(x)
        return x