import torch
from torch import nn
import math



class LocalTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, window_size=512, causal=True, look_backward=1, look_forward=0):
        super(LocalTransformerEncoderLayer, self).__init__()
        self.heads = nn.ModuleList([
            LocalAttention(
                dim=d_model // nhead,  # Ensure the dimension per head is correct
                window_size=window_size,
                causal=causal,
                look_backward=look_backward,
                look_forward=look_forward,
                dropout=dropout
            ) for _ in range(nhead)
        ])
        self.nhead = nhead
        self.d_model = d_model
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        batch_size, seq_length, _ = src.size()
        head_dim = self.d_model // self.nhead

        # Compute queries, keys, and values
        q = self.query_linear(src).view(batch_size, seq_length, self.nhead, head_dim)
        k = self.key_linear(src).view(batch_size, seq_length, self.nhead, head_dim)
        v = self.value_linear(src).view(batch_size, seq_length, self.nhead, head_dim)

        # Apply local attention to each head
        src = torch.cat([self.heads[i](q[:, :, i, :], k[:, :, i, :], v[:, :, i, :]) for i in range(self.nhead)], dim=-1)

        # Combine the heads' outputs
        src = src.view(batch_size, seq_length, self.d_model)

        # Feedforward network
        src2 = self.dropout1(src)
        src = src + src2
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src