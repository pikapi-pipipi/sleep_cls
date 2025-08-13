import math
import torch
import torch.nn as nn
import numpy as np

    
class Classifier(nn.Module):
    def __init__(self, config, pool='attn'):
        super(Classifier, self).__init__()
        self.model_dim = config['classifier']['model_dim']
        self.pool = pool
        self.cfg = config['classifier']

        if config['classifier']['dropout']:
            self.dropout = nn.Dropout(p=0.1)
        
        if pool == 'attn':
            self.w_ha = nn.Linear(self.model_dim, self.model_dim, bias=True)
            self.w_at = nn.Linear(self.model_dim, 1, bias=False)
        self.fc = nn.Linear(self.model_dim, self.cfg['num_classes'])
    
    def forward(self, x):
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'last':
            x = x[:, -1]
        elif self.pool == 'attn':
            a_states = torch.tanh(self.w_ha(x))
            alpha = torch.softmax(self.w_at(a_states), dim=1).view(x.size(0), 1, x.size(1))
            x = torch.bmm(alpha, a_states).view(x.size(0), -1)
        elif self.pool == None:
            x = x
        else:
            raise NotImplementedError
        if self.cfg['dropout']:
            x = self.dropout(x)
        x = self.fc(x)
        
        return x
        


