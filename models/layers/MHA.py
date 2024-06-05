from dataclasses import dataclass

import torch
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

@dataclass
class MHAConfig:
    d_model: int # D
    n_layers: int = 1
    nhead: int = 1
    dropout: float = 0.0
    batch_first: bool = True
    rms_norm_eps: float = 1e-5


class MHA(nn.Module):
    """
    Multi-Head Attention
    """
    def __init__(self, config: MHAConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, config: MHAConfig):
        super().__init__()

        self.mixer = MHABlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):

        output = self.dropout(self.mixer(self.norm(x))) + x
        return output
    

    
class MHABlock(nn.Module):
    def __init__(self, config: MHAConfig):
        super().__init__()

        self.config = config

        self.self_attn = MultiheadAttention(config.d_model, 
                                            config.nhead, 
                                            dropout=config.dropout, 
                                            batch_first=config.batch_first)

    def forward(self, x):
        output = self.self_attn(x, x, x)[0]
        return output
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
    

