import torch.nn as nn
import torch.nn.functional as F

from .eegEncoder import EEGEncoder
from .hemodynamicEncoder import HemodynamicEncoder
from .ppgEncoder import PPGEncoder
from .eogEncoder import EOGEncoder
from .modalFusion import ModalFusion
from .agentTransformer import AgentTransformer

from .classifiers import Classifier
import os
import torch


last_chn_dict = {
    'SleePyCo': 128,
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
            self.ppg2EEGFusion = ModalFusion(self.cfg)
            if self.multimodal[2] or self.multimodal[1]:
                self.fnirs2EEGFusion = ModalFusion(self.cfg)
            if self.bb_cfg['dropout']:
                self.ppgDropout = nn.Dropout(p=0.1)
        if self.multimodal[4]:
            self.eogEncoder = EOGEncoder(self.cfg)
            self.eogFusion = ModalFusion(self.cfg)
            if self.bb_cfg['dropout']:
                self.eogDropout = nn.Dropout(p=0.1)
        
        self.SeqAttn = AgentTransformer(config, 8, 4, 
                                        pool_size_rate=8, 
                                        seq_len=config['Transformer']['eeg_pos_enc']['seq_len'], 
                                        pos_enc_dropout=config['Transformer']['eeg_pos_enc']['dropout'])

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
            
            if self.local_rank == 0:
                print('[INFO] Number of params of sequence attention layer: ', sum(p.numel() for p in self.SeqAttn.parameters() if p.requires_grad))

        elif self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
            self.classifier = Classifier(self.cfg)
            if self.local_rank == 0:
                print('[INFO] Number of params of sequence attention layer: ', sum(p.numel() for p in self.SeqAttn.parameters() if p.requires_grad))
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
        if self.bb_cfg['dropout']:
            eeg_features = self.eegDropout(eeg_features)
        
        if eog is not None:
            eog_feature = self.eogEncoder(eog)
            if self.bb_cfg['dropout']:
                eog_feature = self.eogDropout(eog_feature)
            eog_feature = eog_feature.transpose(1, 2)
        if ppg is not None:
            ppg_feature = self.ppgEncoder(ppg)
            if self.bb_cfg['dropout']:
                ppg_feature = self.ppgDropout(ppg_feature)
            ppg_feature = ppg_feature.transpose(1, 2)

        if hbo is not None:
            hbo_feature = self.hboEncoder(hbo)
            if self.bb_cfg['dropout']:
                hbo_feature = self.hboDropout(hbo_feature)
            hbo_feature = hbo_feature.transpose(1, 2)
        if hb is not None:
            hb_feature = self.hbEncoder(hb)
            if self.bb_cfg['dropout']:
                hb_feature = self.hbDropout(hb_feature)
            hb_feature = hb_feature.transpose(1, 2)    
                
        # 3 conditions
        fnirs_feature = None
        if hbo is not None and hb is not None:
            fnirs_feature = self.hbFusion(hbo_feature, hb_feature)
        elif hbo is not None and hb is None:
            fnirs_feature = hbo_feature
        elif hbo is None and hb is not None:
            fnirs_feature = hb_feature
        
        if fnirs_feature is not None:
            if ppg is not None:
                fnirs_feature = self.hboFusion(fnirs_feature, ppg_feature)

        outputs = []

        if self.training_mode == 'pretrain':
            for eeg_feature in eeg_features:
                eeg_feature = eeg_feature.transpose(1, 2)
                if eog is not None:
                    eeg_feature = self.eogFusion(eeg_feature, eog_feature)
                if ppg is not None:
                    eeg_feature = self.ppg2EEGFusion(eeg_feature, ppg_feature)
                if fnirs_feature is not None:
                    if ppg is not None:
                        eeg_feature = self.fnirs2EEGFusion(eeg_feature, fnirs_feature)
                    else:
                        if hbo is not None:
                            eeg_feature = self.hboFusion(eeg_feature, fnirs_feature)
                        else:
                            eeg_feature = self.hbFusion(eeg_feature, fnirs_feature)
                feature = self.SeqAttn(eeg_feature)
                outputs.append(F.normalize(self.head(feature.transpose(1, 2))))
            
        elif self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
            for eeg_feature in eeg_features:
                eeg_feature = eeg_feature.transpose(1, 2)
                if eog is not None:
                    eeg_feature = self.eogFusion(eeg_feature, eog_feature)
                if ppg is not None:
                    eeg_feature = self.ppg2EEGFusion(eeg_feature, ppg_feature)
                if fnirs_feature is not None:
                    if ppg is not None:
                        eeg_feature = self.fnirs2EEGFusion(eeg_feature, fnirs_feature)
                    else:
                        eeg_feature = self.hboFusion(eeg_feature, fnirs_feature)
                feature = self.SeqAttn(eeg_feature)
                outputs.append(self.classifier(feature))
        else:
            raise NotImplementedError

        return outputs
    

