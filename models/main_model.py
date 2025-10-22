import torch.nn as nn
import torch.nn.functional as F

from .eegEncoder import EEGEncoder
from .hemodynamicEncoder import HemodynamicEncoder
from .ppgEncoder import PPGEncoder
from .modalFusion import ModalFusion
from .agentTransformer import PositionalEncoding, AgentTransformer

from .classifiers import Classifier
import os
import torch


last_chn_dict = {
    'SleePyCo': 128,
}

class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.cfg = config
        self.multimodal = self.cfg['dataset']['multimodal']
        self.bb_cfg = config['backbone']
        
        self.eegEncoder = EEGEncoder(self.cfg)
        if self.bb_cfg['dropout']:
            self.eegDropout = nn.Dropout(p=0.1)
        if self.multimodal[1] and self.multimodal[2]:
            self.hboEncoder = HemodynamicEncoder(self.cfg)
            self.hboFusion = ModalFusion(self.cfg)
            if self.bb_cfg['dropout']:
                self.hboDropout = nn.Dropout(p=0.1)
            self.hbEncoder = HemodynamicEncoder(self.cfg)
            self.hbFusion = ModalFusion(self.cfg)
            if self.bb_cfg['dropout']:
                self.hbDropout = nn.Dropout(p=0.1)
            self.hbo_hb_fc = nn.Linear(self.cfg['feature_pyramid']['dim'] * 2, self.cfg['feature_pyramid']['dim'])
            self.fnirsFusion = ModalFusion(self.cfg)
        if self.multimodal[3]:
            self.ppgEncoder = PPGEncoder(self.cfg)
            self.ppgFusion = ModalFusion(self.cfg)
            if self.bb_cfg['dropout']:
                self.ppgDropout = nn.Dropout(p=0.1)
                
        if self.multimodal[1] or self.multimodal[2] or self.multimodal[3]:
            self.Ffc = nn.Linear(self.cfg['feature_pyramid']['dim'] * (sum(self.multimodal[2:4])), last_chn_dict[config['backbone']['name']])
            self.Efc = nn.Linear(self.cfg['feature_pyramid']['dim'] * (sum(self.multimodal[2:4])+1), last_chn_dict[config['backbone']['name']])
        # self.sequence_modeling = ModalFusion(self.cfg)

    def forward(self, eeg=None, hbo=None, hb=None, ppg=None):

        eeg_features = self.eegEncoder(eeg)
        fnirs_main_feature = []
        
        if ppg is not None:
            ppg_feature = self.ppgEncoder(ppg)
            if self.bb_cfg['dropout']:
                ppg_feature = self.ppgDropout(ppg_feature)
            fnirs_main_feature.append(ppg_feature)

        if hbo is not None:
            hbo_feature = self.hboEncoder(hbo)
            if self.bb_cfg['dropout']:
                hbo_feature = self.hboDropout(hbo_feature)
                
        if hb is not None:
            hb_feature = self.hbEncoder(hb)
            if self.bb_cfg['dropout']:
                hb_feature = self.hbDropout(hb_feature)    

        if self.multimodal[1] and self.multimodal[2]:
            fnirs_feature = self.hbo_hb_fc(torch.concat([
                self.hboFusion(hbo_feature, hb_feature),
                self.hbFusion(hb_feature, hbo_feature)
                ], dim=2))
            fnirs_main_feature.append(fnirs_feature)

        outputs = []
        
        if len(fnirs_main_feature) > 1:
            fnirs_main_feature = self.Ffc(torch.concat(fnirs_main_feature, dim=2))
        elif len(fnirs_main_feature) == 1:
            fnirs_main_feature = self.Ffc(fnirs_main_feature[0])
            
        for eeg_feature in eeg_features:
            if self.bb_cfg['dropout']:
                eeg_feature = self.eegDropout(eeg_feature)
            eeg_fusion_feature = [eeg_feature]
            if ppg is not None:
                eeg_fusion_feature.append(self.ppgFusion(eeg_feature, ppg_feature))
            # if hbo is not None:
            #     eeg_fusion_feature.append(self.hboFusion(eeg_feature, hbo_feature))
            # if hb is not None:
            #     eeg_fusion_feature.append(self.hbFusion(eeg_feature, hb_feature))
            if hbo is not None and hb is not None:
                eeg_fusion_feature.append(self.fnirsFusion(eeg_feature, fnirs_feature))
            if self.multimodal[1] or self.multimodal[2] or self.multimodal[3]:
                eeg_feature = torch.concat(eeg_fusion_feature, dim=2)
                eeg_feature = self.Efc(eeg_feature)
                eeg_feature = torch.concat([eeg_feature, fnirs_main_feature], dim=1)
            # eeg_feature = self.sequence_modeling(eeg_feature, eeg_feature)
            outputs.append(eeg_feature)
        return outputs

class MainModel(nn.Module):
    
    def __init__(self, config):

        super(MainModel, self).__init__()

        self.cfg = config
        self.multimodal = self.cfg['dataset']['multimodal']
        self.training_mode = config['training_params']['mode']
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.encoder = Encoder(self.cfg)


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

    def forward(self, eeg=None, hbo=None, hb=None, ppg=None):
        # x: (batch_size, seq_len, n_samples)
        # bz = eeg.size(0)
        # seq_len = self.cfg['dataset']['seq_len']
        outputs = []
        eeg_features = self.encoder(eeg, hbo, hb, ppg)
        for eeg_feature in eeg_features:
            if self.training_mode == 'pretrain':
                    outputs.append(F.normalize(self.head(eeg_feature.transpose(1, 2))))
                
            elif self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
                    outputs.append(self.classifier(eeg_feature))
            else:
                raise NotImplementedError

        return outputs
    

