import os
import glob
import torch
import numpy as np
from transform import *
from torch.utils.data import Dataset

class EEGDataLoader(Dataset):

    def __init__(self, config, fold, set='train'):

        self.set = set
        self.fold = fold

        self.eeg_sr = 100 
        self.hbo_sr = 25
        self.hb_sr = 25
        self.ppg_sr = 25
        self.eog_sr = 100
        self.dset_cfg = config['dataset']
        self.multimodal = config['dataset']['multimodal']
        
        self.root_dir = self.dset_cfg['root_dir']
        self.dset_name = self.dset_cfg['name']
        self.num_splits = self.dset_cfg['num_splits']
        self.eeg_channel = self.dset_cfg['eeg_channel']
        
        self.seq_len = self.dset_cfg['seq_len']
        self.target_idx = self.dset_cfg['target_idx']
        
        self.training_mode = config['training_params']['mode']

        self.dataset_path = os.path.join(self.root_dir, 'dset', self.dset_name, 'npz')
        self.eeg, self.hbo, self.hb, self.ppg, self.eog, self.labels, self.score, self.epochs = self.split_dataset()


        self.transform = Compose(
            transforms=[
                RandomAmplitudeScale(),
                RandomTimeShift(),
                RandomDCShift(),
                RandomZeroMasking(),
                RandomAdditiveGaussianNoise(),
                RandomBandStopFilter(),
            ],
            mode='full'
        )
        self.two_transform = TwoTransform(self.transform)
        
        # self.fnirs_transform = TwoTransform(ChannelShuffle())
        self.fnirs_transform = TwoTransform(Compose(transforms=[ChannelShuffle(),
                                                                FnirsRandomAmplitudeScale()]))

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        eeg_n_sample = 30 * self.eeg_sr * self.seq_len
        hbo_n_sample = 30 * self.hbo_sr * self.seq_len
        hb_n_sample = 30 * self.hb_sr * self.seq_len
        ppg_n_sample = 30 * self.ppg_sr * self.seq_len
        eog_n_sample = 30 * self.eog_sr * self.seq_len
        score_n_sample = 6 * self.seq_len
        file_idx, idx, seq_len = self.epochs[idx]
        eegs = self.eeg[file_idx][idx:idx+seq_len]

        # if self.score:
        #     scores = self.score[file_idx][idx:idx+seq_len]
        #     scores = np.array(scores).reshape(-1, score_n_sample) # (n_modalities, seq_len)
        #     scores = np.repeat(scores, repeats=125, axis=-1)
        
        hbos, hbs, ppgs, eogs = None, None, None, None
        if self.hbo:
            hbos = self.hbo[file_idx][idx:idx+seq_len]
        if self.hb:
            hbs = self.hb[file_idx][idx:idx+seq_len]
        if self.ppg:
            ppgs = self.ppg[file_idx][idx:idx+seq_len]
        if self.eog:
            eogs = self.eog[file_idx][idx:idx+seq_len]

        if self.set == 'train':
            if self.training_mode == 'pretrain':
                assert seq_len == self.seq_len
                eegs = eegs.reshape(-1, eeg_n_sample)
                input_a, input_b = self.two_transform(eegs)
                input_a = torch.from_numpy(input_a).float()
                input_b = torch.from_numpy(input_b).float()
                eegs = [input_a, input_b]
                
                if hbos is not None:
                    hbos = hbos.reshape(-1, hbo_n_sample)
                    # hbos = np.stack((hbos, scores), axis=0)
                    # hbos = hbos.reshape(-1, hbo_n_sample)
                    input_a, input_b = self.fnirs_transform(hbos)
                    input_a = torch.from_numpy(input_a).float()
                    input_b = torch.from_numpy(input_b).float()
                    hbos = [input_a, input_b]
                else:
                    hbos = [0]
                    
                if hbs is not None:
                    hbs = hbs.reshape(-1, hb_n_sample)
                    # hbs = np.stack((hbs, scores), axis=0)
                    # hbs = hbs.reshape(-1, hb_n_sample)
                    input_a, input_b = self.fnirs_transform(hbs)
                    input_a = torch.from_numpy(input_a).float()
                    input_b = torch.from_numpy(input_b).float()
                    hbs = [input_a, input_b]
                else:
                    hbs = [0]
                if ppgs is not None:
                    ppgs = ppgs.reshape(-1, ppg_n_sample)
                    # scores = np.repeat(scores, repeats=2, axis=0)
                    # ppgs = np.stack((ppgs, scores), axis=0)
                    # ppgs = ppgs.reshape(-1, ppg_n_sample)
                    input_a, input_b = self.fnirs_transform(ppgs)
                    input_a = torch.from_numpy(input_a).float()
                    input_b = torch.from_numpy(input_b).float()
                    ppgs = [input_a, input_b]
                else:
                    ppgs = [0]
                    
                if eogs is not None:
                    eogs = eogs.reshape(-1, eog_n_sample)
                    eogs = torch.from_numpy(eogs).float()
                    eegs[0] = torch.cat((eegs[0], eogs), dim=0)
                    eegs[1] = torch.cat((eegs[1], eogs), dim=0)
                else:
                    eogs = [0]
                    
            
            elif self.training_mode in ['scratch', 'fullyfinetune', 'freezefinetune']:

                eegs = eegs.reshape(-1, eeg_n_sample)
                eegs = torch.from_numpy(eegs).float()
                if hbos is not None:
                    hbos = hbos.reshape(-1, hbo_n_sample)
                    # hbos = np.stack((hbos, scores), axis=0)
                    # hbos = hbos.reshape(-1, hbo_n_sample)
                    hbos = torch.from_numpy(hbos).float()
                else:
                    hbos = [0]
                if hbs is not None:
                    hbs = hbs.reshape(-1, hb_n_sample)
                    # hbs = np.stack((hbs, scores), axis=0)
                    # hbs = hbs.reshape(-1, hb_n_sample)
                    hbs = torch.from_numpy(hbs).float()
                else:
                    hbs = [0]
                if ppgs is not None:
                    ppgs = ppgs.reshape(-1, ppg_n_sample)
                    # scores = np.repeat(scores, repeats=2, axis=0)
                    # ppgs = np.stack((ppgs, scores), axis=0)
                    # ppgs = ppgs.reshape(-1, ppg_n_sample)
                    ppgs = torch.from_numpy(ppgs).float()
                else:
                    ppgs = [0]
                if eogs is not None:
                    eogs = eogs.reshape(-1, eog_n_sample)
                    eogs = torch.from_numpy(eogs).float()
                    eegs = torch.cat((eegs, eogs), dim=0)
                else:
                    eogs = [0]
                    
            else:
                raise NotImplementedError
            
        else:
            eegs = eegs.reshape(-1, eeg_n_sample)
            eegs = torch.from_numpy(eegs).float()
            if hbos is not None:
                hbos = hbos.reshape(-1, hbo_n_sample)
                # hbos = np.stack((hbos, scores), axis=0)
                # hbos = hbos.reshape(-1, hbo_n_sample)
                hbos = torch.from_numpy(hbos).float()
            else:
                hbos = [0]
            if hbs is not None:
                hbs = hbs.reshape(-1, hb_n_sample)
                # hbs = np.stack((hbs, scores), axis=0)
                # hbs = hbs.reshape(-1, hb_n_sample)
                hbs = torch.from_numpy(hbs).float()
            else:
                hbs = [0]
            if ppgs is not None:
                ppgs = ppgs.reshape(-1, ppg_n_sample)
                # scores = np.repeat(scores, repeats=2, axis=0)
                # ppgs = np.stack((ppgs, scores), axis=0)
                # ppgs = ppgs.reshape(-1, ppg_n_sample)
                ppgs = torch.from_numpy(ppgs).float()
            else:
                ppgs = [0]
                
            if eogs is not None:
                eogs = eogs.reshape(-1, eog_n_sample)
                eogs = torch.from_numpy(eogs).float()
                eegs = torch.cat((eegs, eogs), dim=0)

        labels = self.labels[file_idx][idx:idx+seq_len]
        labels = torch.from_numpy(labels).long()
        labels = labels[self.target_idx]

        return eegs, hbos, hbs, ppgs, labels

    def split_dataset(self):

        file_idx = 0
        eeg, hbo, hb, ppg, eog, labels, epochs = [], [], [], [], [], [], []
        score = []
        data_root = os.path.join(self.dataset_path, self.eeg_channel)
        data_fname_list = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(data_root, '*.npz')))]
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        split_idx_list = np.load(os.path.join('./split_idx', 'idx_{}.npy'.format(self.dset_name)), allow_pickle=True)

        assert len(split_idx_list) == self.num_splits
    
        if self.dset_name == 'EFSleep':
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][1:3])
                if subject_idx in split_idx_list[self.fold-1][self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])
        else:
            raise NameError("dataset '{}' cannot be found.".format(self.dataset))
            
        for data_fname in data_fname_dict[self.set]:
            npz_file = np.load(os.path.join(data_root, data_fname))
            eeg.append(npz_file['eeg'])
            if self.multimodal[1]:
                hbo.append(npz_file['hbo'])
            if self.multimodal[2]:
                hb.append(npz_file['hb'])
            if self.multimodal[3]:
                ppg.append(npz_file['ppg'])
            if self.multimodal[4]:
                eog.append(npz_file['eog'])
            
            labels.append(npz_file['label'])
            if self.multimodal[1] or self.multimodal[2] or self.multimodal[3]:
                score.append(npz_file['score'])  # (n_modalities, seq_len, 8)
            seq_len = self.seq_len
            for i in range(len(npz_file['label']) - seq_len + 1):
                epochs.append([file_idx, i, seq_len])
            file_idx += 1
        if not hbo:
            hbo = None
        if not hb:
            hb = None
        if not ppg:
            ppg = None
        if not eog:
            eog = None

        return eeg, hbo, hb, ppg, eog, labels, score, epochs

if __name__ == '__main__':
    config = {
        'dataset': {
            'root_dir': './',
            'name': 'EFSleep',
            'num_splits': 5,
            'eeg_channel': 'Fpz',
            'seq_len': 10,
            'target_idx': -1,
            'multimodal': [True, True, True, True, True]
        },
        'training_params': {
            'mode': 'pretrain'
        }
    }
    fold = 1
    dset = EEGDataLoader(config, fold, set='train')
    loader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    for i, (eeg, hbo, hb, ppg, labels) in enumerate(loader):
        print(eeg[0].shape)
        print(hbo[0].shape)
        print(hb[0].shape)
        print(ppg[0].shape)
        print(labels[0].shape)
        break