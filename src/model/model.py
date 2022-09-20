from torch import nn
import torch

class AttnDecoder(nn.Module):
    
    def __init__(self, enc_hidden_size, hidden_size, n_out):
        super(AttnDecoder, self).__init__()
        self.dec = nn.GRUCell(input_size=enc_hidden_size, hidden_size = hidden_size)
        self.query_proj = nn.LazyLinear(hidden_size)
        self.key_proj = nn.LazyLinear(hidden_size)
        self.activation = nn.ReLU()
        self.w_v = nn.LazyLinear(1)
        self.linear1 = nn.LazyLinear(500)
        self.linear2 = nn.LazyLinear(200)
        self.linear3 = nn.LazyLinear(100)
        self.linear4 = nn.LazyLinear(50)
        self.linear5 = nn.LazyLinear(10)
        self.linear6 = nn.LazyLinear(n_out)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, hidden_prev, hidden_enc):
        attn_score = self.w_v(torch.tanh(self.query_proj(hidden_prev.unsqueeze(1).repeat(1,hidden_enc.size(1),1))+self.key_proj(hidden_enc)))
        context = torch.bmm(attn_score.transpose(1,2), hidden_enc).squeeze(1)
        h = self.dec(context, hidden_prev)
        
        x = self.linear1(h)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(h)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear4(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear5(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear6(x)
        return x, h



class TimeSeriesForecastingModel(nn.Module):
    
    def __init__(self, len_forecast, n_out):
        super(TimeSeriesForecastingModel, self).__init__()
        self.activation = nn.ReLU()
        self.conv = nn.LazyConv1d(512, 3, 1, padding = 1)
        self.enc = nn.GRU(batch_first = True, input_size = 512, hidden_size = 256, num_layers = 1)
        self.attndec = AttnDecoder(256, 256, n_out)
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten(1)
        self.norm = nn.LazyBatchNorm1d()
        self.len_forecast = len_forecast
        
        
    
    def forward(self, input):
   
        x = self.conv(input.float())
        x = self.activation(x)
        x = self.dropout(x)
        x, h = self.enc(input.float().transpose(1,2))
        
        h_dec = [None] * self.len_forecast
        out = [None] * self.len_forecast
        out[0], h_dec[0] = self.attndec(x[:,-1,:].squeeze(1), x)
        for i in range(1, self.len_forecast):
            out[i], h_dec[i] = self.attndec(h_dec[i-1], x)
        out = torch.stack(out, dim = 1)
        return out.squeeze()