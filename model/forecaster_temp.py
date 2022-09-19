from torch import nn
import torch
from layers import attndecoder, conv1d, encoder, fcc

class ForecasterTemp(nn.Module):
    
    def __init__(self, len_forecast):
        super(ForecasterTemp, self).__init__()
        self.conv1d = conv1d([(512, 3, 1, 1)])
        self.dropout = nn.Dropout(0.2)
        self.encoder = encoder(512, 512, 1)
        self.decoder = attndecoder(512, 256)
        self.fcc = fcc([500, 200, 100, 50, 10, 1])
        self.len_forecast = len_forecast
    def forward(self, x):
        x = self.conv1d(x)
        x = self.dropout(x)
        x, h = self.encoder(x.transpose(1,2))

        h_dec = [None] * self.len_forecast
        out = [None] * self.len_forecast
        out[0], h_dec[0] = self.attndec(x[:,-1,:].squeeze(1), x)
        for i in range(1, self.len_forecast):
            out[i], h_dec[i] = self.attndec(h_dec[i-1], x)
        out = torch.stack(out, dim = 1)
        return out.squeeze()