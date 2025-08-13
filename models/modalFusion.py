import math
import torch
import torch.nn as nn
import numpy as np

class ModalFusion(nn.Module):
    def __init__(self, config):
        super(ModalFusion, self).__init__()
        self.config = config
        self.model_dim = config['classifier']['model_dim']
        self.Q = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.K = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.V = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.fc = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.dropout = nn.Dropout(p=0.1 if config['classifier']['dropout'] else 0.0)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x1, x2):
        q = self.Q(x1)
        k = self.K(x2)
        v = self.V(x2)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.model_dim)
        attention_weights = self.softmax(attention_scores)
        
        context = torch.matmul(attention_weights, v)
        context = self.fc(context)
        
        if self.config['classifier']['dropout']:
            context = self.dropout(context)
        
        context = context + x1  # Residual connection
        
        return context