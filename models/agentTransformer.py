import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.cuda.amp import autocast
import torch.utils.checkpoint as checkpoint
import numpy as np

class PositionalEncoding(nn.Module):
    
    def __init__(self, config, in_features, out_features, max_len=5000, dropout=False):
        super(PositionalEncoding, self).__init__()
        self.num_scales = config['feature_pyramid']['num_scales']
        
        self.dropout = dropout
        if dropout is not False:
            self.dropout = nn.Dropout(p=dropout)
        
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.act_fn = nn.PReLU()
        
        self.max_len = max_len

        pe = torch.zeros(self.max_len, out_features)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, out_features, 2).float() * (-math.log(10000.0) / out_features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.act_fn(self.fc(x))
        if self.num_scales > 1:
            hop = self.max_len // x.size(1)
            pe = self.pe[:, hop//2::hop]
        else:
            pe = self.pe

        if pe.shape[1] != x.size(1):
            pe = pe[:, :x.size(0)]
        x = x + pe

        if self.dropout is not False:
            x = self.dropout(x)

        return x

class AgentAttention(nn.Module):
    def __init__(self, d_model, nheads, pool_size_rate, qkv_bias=True, dropout=0, head_strides=None):
        super(AgentAttention, self).__init__()
        self.dim = d_model
        self.nheads = nheads
        self.dropout = nn.Dropout(dropout)
        head_dim = d_model // nheads
        self.scale =  head_dim ** -0.5
        self.q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.kv = nn.Linear(d_model, d_model * 2, bias=qkv_bias)
        self.proj = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.pool_size_rate = pool_size_rate
        self.pool = nn.MaxPool1d(pool_size_rate, pool_size_rate)
        # Multi-scale head groups for MaxFormer
        if head_strides is None or len(head_strides) == 0:
            self.head_strides = [1, max(1, pool_size_rate), max(1, pool_size_rate * 2)]
        else:
            self.head_strides = [max(1, int(s)) for s in head_strides]
        self.head_groups = self._build_head_groups()

    def _build_head_groups(self):
        num_groups = len(self.head_strides)
        base = self.nheads // num_groups
        rem = self.nheads % num_groups
        groups = []
        start = 0
        for i in range(num_groups):
            size = base + (1 if i < rem else 0)
            end = start + size
            groups.append(list(range(start, end)))
            start = end
        return groups

    def _pool_tokens(self, x, stride):
        if stride <= 1:
            return x
        b, n, d = x.size()
        x_t = x.transpose(1, 2)
        pad_size = (stride - (n % stride)) % stride
        if pad_size > 0:
            left_pad = pad_size // 2
            right_pad = pad_size - left_pad
            x_t = F.pad(x_t, (left_pad, right_pad), mode='constant', value=0.0)
        pooled = F.max_pool1d(x_t, kernel_size=stride, stride=stride)
        return pooled.transpose(1, 2)
        
    def forward(self, x):
        b, n, d = x.size()
        head_dim = d // self.nheads
        q = self.q(x).reshape(b, n, self.nheads, head_dim).permute(0, 2, 1, 3)
        multi_scale_out = torch.zeros_like(q)

        for head_ids, stride in zip(self.head_groups, self.head_strides):
            if len(head_ids) == 0:
                continue
            agent_tokens = self._pool_tokens(x, stride)
            n_agent = agent_tokens.size(1)
            k, v = self.kv(agent_tokens).view(b, n_agent, 2, d).permute(2, 0, 1, 3)
            k = k.reshape(b, n_agent, self.nheads, head_dim).permute(0, 2, 1, 3)
            v = v.reshape(b, n_agent, self.nheads, head_dim).permute(0, 2, 1, 3)

            q_group = q[:, head_ids, :, :]
            k_group = k[:, head_ids, :, :]
            v_group = v[:, head_ids, :, :]
            q_attn = self.softmax((q_group * self.scale) @ k_group.transpose(-2, -1))
            q_attn = self.dropout(q_attn)
            multi_scale_out[:, head_ids, :, :] = q_attn @ v_group

        x = multi_scale_out.transpose(1, 2).reshape(b, n, d)
        
        x = self.proj(x)
        
        return x

class AgentTransformerLayer(nn.Module):
    def __init__(self, d_model, nheads, pool_size_rate, dim_feedforward=512, dropout=0, head_strides=None):
        super(AgentTransformerLayer, self).__init__()
        self.dim = d_model
        self.nheads = nheads
        self.feedforward_dim = dim_feedforward
        self.dropout = dropout
        
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.PReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        self.attn = AgentAttention(d_model, nheads, pool_size_rate, dropout=dropout, head_strides=head_strides)
        
    def forward(self, x):
        assert x.size(-1) == self.dim
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.dropout2(self.feedforward(self.norm2(x)))
        return x
        
        
class AgentTransformerEncoder(nn.Module):
    def __init__(self, d_model, nheads, num_layers, pool_size_rate, dim_feedforward=512, dropout=0, head_strides=None):
        super(AgentTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            AgentTransformerLayer(d_model, nheads, pool_size_rate, dim_feedforward, dropout, head_strides=head_strides)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    

class AgentTransformer(nn.Module):

    def __init__(self, config, nheads, num_encoder_layers, pool_size_rate, seq_len, pos_enc_dropout):

        super(AgentTransformer, self).__init__()
        self.training_mode = config['training_params']['mode']
        self.cfg = config['Transformer']
        self.model_dim = self.cfg['model_dim']
        self.feedforward_dim = self.cfg['feedforward_dim']

        self.in_features = config['feature_pyramid']['dim']
        self.out_features = self.cfg['model_dim']
        self.maxformer_head_strides = self.cfg.get('maxformer_head_strides', None)
        
        self.pos_encoding = PositionalEncoding(config, self.in_features, self.out_features, seq_len, pos_enc_dropout)
        self.layers = AgentTransformerEncoder(
            self.model_dim, nheads, num_encoder_layers, pool_size_rate,
            self.feedforward_dim, self.cfg['dropout'], head_strides=self.maxformer_head_strides
        )


    def forward(self, x):
        # x: (batch_size, seq_len, n_samples)
        x = self.pos_encoding(x)
        x = self.layers(x)
        
        return x
    
if __name__ == '__main__':
    x = torch.randn(32, 48, 128)
    config = {
        'Transformer': {
            'model_dim': 128,
            'feedforward_dim': 512,
            'dropout': 0.1,
            'num_classes': 5,
            'pos_enc': {
                'seq_len': 1200,
                'dropout': 0.1
            }
        },
        'feature_pyramid': {
            'dim': 128,
            'num_scales': 3
        },
        'training_params': {
            'mode': 'pretrain'
        }
    }
    model = AgentTransformer(config, 8, 6, pool_size_rate=6, seq_len=12, pos_enc_dropout=0.1)
    out = model(x)
    print(out.shape)
    