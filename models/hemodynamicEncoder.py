import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class HemodynamicBackbone(nn.Module):
    
    def __init__(self, config, kernel_size=3):
        super(HemodynamicBackbone, self).__init__()

        self.training_mode = config['training_params']['mode']

        # architecture
        self.init_layer = self.make_layers(in_channels=8, out_channels=64, n_layers=2, maxpool_size=None, kernel_size=kernel_size, first=True)
        self.layer1 = self.make_layers(in_channels=64, out_channels=128, n_layers=2, maxpool_size=5, kernel_size=kernel_size)
        self.layer2 = self.make_layers(in_channels=128, out_channels=192, n_layers=3, maxpool_size=5, kernel_size=kernel_size)
        self.layer3 = self.make_layers(in_channels=192, out_channels=256, n_layers=3, maxpool_size=5, kernel_size=kernel_size)
        self.layer4 = self.make_layers(in_channels=256, out_channels=256, n_layers=3, maxpool_size=5, kernel_size=kernel_size)

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

class HemodynamicEncoder(nn.Module):
    def __init__(self, config):
        super(HemodynamicEncoder, self).__init__()
        self.cfg = config
        self.backbone = HemodynamicBackbone(config)
        self.latent_layer = LatentEncoder(self.cfg)
        
    def forward(self, x):
        return self.latent_layer(self.backbone(x))


class LatentEncoder(nn.Module):
    def __init__(self, config):
        super(LatentEncoder, self).__init__()
        self.fp_dim = config['feature_pyramid']['dim']
        self.num_scales = config['feature_pyramid']['num_scales']
        self.conv = nn.Conv1d(256, self.fp_dim, 1, 1, 0)
            
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,l = y.size()
        return F.interpolate(x, size=l, mode='linear', align_corners=True) + y
    
    def forward(self, x):
        return self.conv(x)
    
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
    model = HemodynamicEncoder(config)
    x = torch.randn(1, 8, 7500)
    out = model(x)
    print(out.shape)