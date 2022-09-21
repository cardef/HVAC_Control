from torch import nn
import torch
from .layers import attndecoder, conv1d, encoder, fcc

class ForecasterEnergy(nn.Module):
    
    def __init__(self, len_forecast):
        super(ForecasterEnergy, self).__init__()
        self.conv1d = conv1d.Conv1d([(512, 3, 1, 1)])
        self.dropout = nn.Dropout(0.2)
        self.encoder = encoder.Encoder(512, 256, 1)
        self.decoder = attndecoder.AttnDecoder(256, 256)
        self.fcc = fcc.FCC([200, 100, 50, 10], 0.2)
        self.output = nn.LazyLinear(1)
        self.len_forecast = len_forecast
    def forward(self, x):
        x = self.conv1d(x)
        x = self.dropout(x)
        x, h = self.encoder(x.transpose(1,2))

        out = [None] * self.len_forecast
        out[0] = self.decoder(x[:,-1,:].squeeze(1), x)
        for i in range(1, self.len_forecast):
            out[i] = self.decoder(out[i-1], x)
        out = torch.stack(out, dim = 1)
        out = self.fcc(out)
        out = self.output(out)
        return out