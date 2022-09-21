from torch import nn
import torch
from .layers import attndecoder, conv1d, encoder, fcc

class ForecasterTemp(nn.Module):
    
    def __init__(self, len_forecast, col_out):
        super(ForecasterTemp, self).__init__()
        self.conv1d = conv1d.Conv1d([(1024, 3, 1, 1), (512, 5, 1, 1)])
        self.dropout = nn.Dropout(0.3)
        self.encoder = encoder.Encoder(512, 512, 1)
        self.decoder = attndecoder.AttnDecoder(512, 512)
        self.fcc = fcc.FCC([1000, 500, 250, 100], 0.3)
        self.output = nn.LazyLinear(col_out)
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