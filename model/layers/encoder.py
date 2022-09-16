from torch import nn
import torch

class Encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.enc = nn.GRU(batch_first = True, input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)
        
    def forward(self, input):
   
        x, h = self.enc(input)
  
        return x, h