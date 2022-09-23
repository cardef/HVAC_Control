from torch import nn
import torch

class AttnDecoder(nn.Module):
    
    def __init__(self, enc_hidden_size, hidden_size):
        super(AttnDecoder, self).__init__()
        self.dec = nn.GRUCell(input_size=enc_hidden_size, hidden_size = hidden_size)
        self.query_proj = nn.LazyLinear(hidden_size)
        self.key_proj = nn.LazyLinear(hidden_size)
        self.w_v = nn.LazyLinear(1)
        
    def forward(self, hidden_prev, hidden_enc):
        attn_score = self.w_v(torch.tanh(self.query_proj(hidden_prev.unsqueeze(1).repeat(1,hidden_enc.size(1),1))+self.key_proj(hidden_enc)))
        context = torch.bmm(attn_score.transpose(1,2), hidden_enc).squeeze(1)
        h = self.dec(context, hidden_prev)

        return h