import math
import torch
import torch.nn as nn

class luong_gate_attention(nn.Module):

    def __init__(self, hidden_size, emb_size, prob=0.2):
        super(luong_gate_attention, self).__init__()
        self.hidden_size, self.emb_size = hidden_size, emb_size
        self.linear_in = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Dropout(p=prob))
        self.feed = nn.Sequential(nn.Linear(2*hidden_size, hidden_size), nn.SELU(
        ), nn.Dropout(p=prob), nn.Linear(hidden_size, 1), nn.Tanh(), nn.Dropout(p=prob))
        self.linear_out = nn.Sequential(nn.Linear(2*hidden_size, hidden_size), nn.SELU(), nn.Dropout(
            p=prob), nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.softmax = nn.Softmax(dim=1)

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, c_t):
        tau = torch.exp(self.feed(torch.cat([h, c_t], 1)))
        gamma_h = self.linear_in(h).unsqueeze(2)
        weights = torch.bmm(self.context, gamma_h).squeeze(2) / tau
        weights = self.softmax(weights)
        c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1)
        output = self.linear_out(torch.cat([h, c_t], 1))
        return output, weights, c_t