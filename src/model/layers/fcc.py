from torch import nn
import torch

class FCC(nn.Module):
    def __init__(self, list_layers, p_dropout):
        super(FCC, self).__init__()
        self.fcc = nn.ModuleList([nn.LazyLinear(layer) for layer in list_layers])
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p_dropout)
    def forward(self, x):
        for layer in self.fcc:
            x = self.activation(x)
            x = layer(x)
            x = self.dropout(x)            
        return x