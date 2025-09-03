import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .agentTransformer import PositionalEncoding, AgentTransformer

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

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, avgpool_size, kernel_size=3, first=False):
        super(Bottleneck, self).__init__()
        self.maxpool = MaxPool1d(avgpool_size) if not first else None
        layers = []
        for i in range(n_layers):
            conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
            layers += [conv1d, nn.BatchNorm1d(out_channels)]
            layers += [nn.PReLU()]
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        if self.maxpool:
            x = self.maxpool(x)
        x = self.layers(x)
        return x

class PPGBackbone(nn.Module):
    
    def __init__(self, config, kernel_size=3):
        super(PPGBackbone, self).__init__()

        self.training_mode = config['training_params']['mode']

        # architecture
        self.init_layer = self.make_layers(in_channels=2, out_channels=32, n_layers=1, maxpool_size=None, kernel_size=kernel_size, first=True)
        self.layer1 = self.make_layers(in_channels=32, out_channels=64, n_layers=1, maxpool_size=5, kernel_size=kernel_size)
        self.layer2 = self.make_layers(in_channels=64, out_channels=96, n_layers=1, maxpool_size=5, kernel_size=kernel_size)
        self.layer3 = self.make_layers(in_channels=96, out_channels=128, n_layers=1, maxpool_size=5, kernel_size=kernel_size)
        self.layer4 = self.make_layers(in_channels=128, out_channels=128, n_layers=1, maxpool_size=5, kernel_size=kernel_size)

        if config['backbone']['init_weights']:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, in_channels, out_channels, n_layers, maxpool_size, kernel_size=3, first=False):
        return Bottleneck(in_channels, out_channels, n_layers, maxpool_size, kernel_size=kernel_size, first=first)

    def forward(self, x):
        x = self.init_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

class PPGEncoder(nn.Module):
    def __init__(self, config):
        super(PPGEncoder, self).__init__()
        self.cfg = config
        self.backbone = PPGBackbone(config)
        self.SeqAttn = AgentTransformer(config, 8, 4, pool_size_rate=1, 
                                        seq_len=config['Transformer']['ppg_pos_enc']['seq_len'], 
                                        pos_enc_dropout=config['Transformer']['ppg_pos_enc']['dropout'])

    def forward(self, x):
        return self.SeqAttn(self.backbone(x).transpose(1, 2))

    
if __name__ == '__main__':
    config = {
        'feature_pyramid': {
            'dim': 128,
            'num_scales': 3
        },
        'backbone': {
            'init_weights': True
        },
        'training_params': {
            'mode': 'pretrain'
        },
    }
    model = PPGEncoder(config)
    x = torch.randn(1, 1, 7500)
    out = model(x)
    print(out.shape)