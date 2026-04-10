import math
import torch
import torch.nn as nn
import numpy as np


class ModalCat(nn.Module):
    def __init__(self, config):
        super(ModalCat, self).__init__()
        self.config = config
        self.model_dim = config['Transformer']['model_dim']
        self.feedforward = nn.Linear(self.model_dim, self.model_dim, bias=False)
    
    def forward(self, x1, x2):
        x = torch.concat([x1,x2], dim=1)
        return self.feedforward(x)


class ModalFusion(nn.Module):
    def __init__(self, config):
        super(ModalFusion, self).__init__()
        self.config = config
        self.model_dim = config['Transformer']['model_dim']
        self.forward_dim = config['Transformer']['feedforward_dim']
        self.Q = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.K = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.V = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.dropout = nn.Dropout(p=0.1 if config['Transformer']['dropout'] else 0.0)
        self.softmax = nn.Softmax(dim=-1)
        self.normx1 = nn.LayerNorm(self.model_dim)
        self.normx2 = nn.LayerNorm(self.model_dim)
        self.normfc = nn.LayerNorm(self.model_dim)
        self.normffn= nn.LayerNorm(self.model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(self.model_dim, self.forward_dim),
            nn.PReLU(),
            nn.Linear(self.forward_dim, self.model_dim),
        )
        
    def forward(self, x1, x2):
        x1 = self.normx1(x1)
        x2 = self.normx2(x2)
        q = self.Q(x1)
        k = self.K(x2)
        v = self.V(x2)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.model_dim)
        attention_weights = self.softmax(attention_scores)
        
        context = torch.matmul(attention_weights, v)
        
        if self.config['Transformer']['dropout']:
            context = self.dropout(context)

        context = self.normfc(context + x1)  # Residual connection

        context = self.feedforward(context) + context

        return self.normffn(context)