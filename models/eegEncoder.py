import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, maxpool_size, kernel_size=3, first=False):
        super(Bottleneck, self).__init__()
        self.maxpool = MaxPool1d(maxpool_size) if not first else None
    
        layers = []
        for i in range(n_layers):
            conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
            layers += [conv1d, nn.BatchNorm1d(out_channels)]
            if i == n_layers - 1:
                layers += [ChannelGate(in_channels)]
            layers += [nn.PReLU()]
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        if self.maxpool:
            x = self.maxpool(x)
        x = self.layers(x)
        return x

class EEGBackbone(nn.Module):
    
    def __init__(self, config, kernel_size=3):
        super(EEGBackbone, self).__init__()

        self.training_mode = config['training_params']['mode']

        # architecture
        self.init_layer = self.make_layers(in_channels=1, out_channels=64, n_layers=2, maxpool_size=None, kernel_size=kernel_size, first=True)
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
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c5, c4, c3

class EEGEncoder(nn.Module):
    def __init__(self, config):
        super(EEGEncoder, self).__init__()
        self.cfg = config
        self.backbone = EEGBackbone(config)
        self.latent_layer = LatentEncoder(self.cfg)

    def forward(self, x):
        return self.latent_layer(self.backbone(x))  

    

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


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = F.max_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class LatentEncoder(nn.Module):
    def __init__(self, config):
        super(LatentEncoder, self).__init__()
        self.fp_dim = config['feature_pyramid']['dim']
        self.num_scales = config['feature_pyramid']['num_scales']
        self.conv_c5 = nn.Conv1d(256, self.fp_dim, 1, 1, 0)

        if self.num_scales > 1:
            self.conv_c4 = nn.Conv1d(256, self.fp_dim, 1, 1, 0)
        
        if self.num_scales > 2:
            self.conv_c3 = nn.Conv1d(192, self.fp_dim, 1, 1, 0)
            
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
        out = []
        c5, c4, c3 = x
        c5 = self.conv_c5(c5)
        out.append(c5)
        if self.num_scales > 1:
            c4 = self.conv_c4(c4)
            out.append(c4)
        if self.num_scales > 2:
            c3 = self.conv_c3(c3)
            out.append(c3)
        
        return out
        
         

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
    model = EEGEncoder(config)
    x = torch.randn(1, 1, 30000)
    out = model(x)
    print(out[0].shape, out[1].shape, out[2].shape)
