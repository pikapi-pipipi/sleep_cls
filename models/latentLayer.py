import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPool1d(nn.Module):
    def __init__(self, maxpool_size):
        super(MaxPool1d, self).__init__()
        self.maxpool_size = maxpool_size
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool_size, stride=maxpool_size)

    def forward(self, x):
        _, _, n_samples = x.size()
        if n_samples % self.maxpool_size != 0:
            pad_size = self.maxpool_size - (n_samples % self.maxpool_size)
            if pad_size % 2 != 0:
                left_pad = pad_size // 2
                right_pad = pad_size // 2 + 1
            else:
                left_pad = pad_size // 2
                right_pad = pad_size // 2
            x = F.pad(x, (left_pad, right_pad), mode='constant')

        x = self.maxpool(x)

        return x
    

class AttnLayer(nn.Module):
    def __init__(self, model_dim, expansion=1):
        super(AttnLayer, self).__init__()
        self.model_dim = model_dim
        self.w_ha = nn.Linear(self.model_dim, self.model_dim * expansion, bias=False)
        self.w_at = nn.Linear(self.model_dim * expansion , 1, bias=False)
    
    def forward(self, x):
        a_states = torch.tanh(self.w_ha(x))
        alpha = torch.softmax(self.w_at(a_states), dim=1).view(x.size(0), 1, x.size(1))
        x = torch.bmm(alpha, a_states).view(x.size(0), -1)
        
        return x



