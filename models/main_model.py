import torch.nn as nn
import torch.nn.functional as F

from .eegEncoder import EEGEncoder
from .hemodynamicEncoder import HemodynamicEncoder
from .ppgEncoder import PPGEncoder
from .eogEncoder import EOGEncoder
from .modalFusion import ModalFusion
from .agentTransformer import PositionalEncoding, AgentTransformer

from .classifiers import Classifier
import os
import torch


last_chn_dict = {
    'SleePyCo': 256,
}


class MainModel(nn.Module):
    
    def __init__(self, config):

        super(MainModel, self).__init__()

        self.cfg = config
        self.multimodal = self.cfg['dataset']['multimodal']
        self.bb_cfg = config['backbone']
        self.training_mode = config['training_params']['mode']
        self.local_rank = int(os.environ["LOCAL_RANK"])

        self.eegEncoder = EEGEncoder(self.cfg)
        if self.bb_cfg['dropout']:
            self.eegDropout = nn.Dropout(p=0.1)
        if self.multimodal[1]:
            self.hboEncoder = HemodynamicEncoder(self.cfg)
            self.hboFusion = ModalFusion(self.cfg)
            if self.bb_cfg['dropout']:
                self.hboDropout = nn.Dropout(p=0.1)
        if self.multimodal[2]:
            self.hbEncoder = HemodynamicEncoder(self.cfg)
            self.hbFusion = ModalFusion(self.cfg)
            if self.bb_cfg['dropout']:
                self.hbDropout = nn.Dropout(p=0.1)
        if self.multimodal[3]:
            self.ppgEncoder = PPGEncoder(self.cfg)
            self.ppgFusion = ModalFusion(self.cfg)
            if self.bb_cfg['dropout']:
                self.ppgDropout = nn.Dropout(p=0.1)
        if self.multimodal[4]:
            self.eogEncoder = EOGEncoder(self.cfg)
            self.eogFusion = ModalFusion(self.cfg)
            if self.bb_cfg['dropout']:
                self.eogDropout = nn.Dropout(p=0.1)
                
        self.fc = nn.Linear(self.cfg['feature_pyramid']['dim'] * sum(self.multimodal), last_chn_dict[config['backbone']['name']])

        if self.training_mode == 'pretrain':
            proj_dim = self.cfg['proj_head']['dim']
            if config['proj_head']['name'] == 'Linear':
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(last_chn_dict[config['backbone']['name']], proj_dim)
                )
            elif config['proj_head']['name'] == 'MLP':
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(last_chn_dict[config['backbone']['name']], proj_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(proj_dim, proj_dim)
                    )

            else:
                raise NotImplementedError('head not supported: {}'.format(config['proj_head']['name']))

        elif self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
            self.classifier = Classifier(self.cfg)
            if self.local_rank == 0:
                print('[INFO] Number of params of classifier: ', sum(p.numel() for p in self.classifier.parameters() if p.requires_grad))
             
        else:
            raise NotImplementedError('head not supported: {}'.format(config['training_params']['mode']))           

    def get_max_len(self, features):
        len_list = []
        for feature in features:
            len_list.append(feature.shape[1])
        
        return max(len_list)

    def forward(self, eeg=None, hbo=None, hb=None, ppg=None, eog=None):
        # x: (batch_size, seq_len, n_samples)
        # bz = eeg.size(0)
        # seq_len = self.cfg['dataset']['seq_len']
        eeg_features = self.eegEncoder(eeg)
        
        if eog is not None:
            eog_feature = self.eogEncoder(eog)
            if self.bb_cfg['dropout']:
                eog_feature = self.eogDropout(eog_feature)
        if ppg is not None:
            ppg_feature = self.ppgEncoder(ppg)
            if self.bb_cfg['dropout']:
                ppg_feature = self.ppgDropout(ppg_feature)

        if hbo is not None:
            hbo_feature = self.hboEncoder(hbo)
            if self.bb_cfg['dropout']:
                hbo_feature = self.hboDropout(hbo_feature)
                
        if hb is not None:
            hb_feature = self.hbEncoder(hb)
            if self.bb_cfg['dropout']:
                hb_feature = self.hbDropout(hb_feature)        

        outputs = []
        for eeg_feature in eeg_features:
            if self.bb_cfg['dropout']:
                eeg_feature = self.eegDropout(eeg_feature)
            eeg_fusion_feature = []
            eeg_fusion_feature.append(eeg_feature)
            if eog is not None:
                eeg_fusion_feature.append(self.eogFusion(eeg_feature, eog_feature))
            if ppg is not None:
                eeg_fusion_feature.append(self.ppgFusion(eeg_feature, ppg_feature))
            if hbo is not None:
                eeg_fusion_feature.append(self.hboFusion(eeg_feature, hbo_feature))
            if hb is not None:
                eeg_fusion_feature.append(self.hbFusion(eeg_feature, hb_feature))
            eeg_feature = torch.concat(eeg_fusion_feature, dim=2)
            eeg_feature = self.fc(eeg_feature)
            if self.training_mode == 'pretrain':
                    outputs.append(F.normalize(self.head(eeg_feature.transpose(1, 2))))
                
            elif self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
                    outputs.append(self.classifier(eeg_feature))
            else:
                raise NotImplementedError

        return outputs
    

