from torch import nn
import torch

class ResEncoder(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(ResEncoder, self).__init__()
        self.enc = nn.GRUCell(input_size=input_size, hidden_size = input_size)
        
    def forward(self, input, hidden):
        
        h = self.enc(input, hidden)
  
        return h + input