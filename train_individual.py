import os
import json
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.utils as nn_utils
import datetime 

from utils import *
from models.main_model import MainModel
import torch.distributed as dist
import gc
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import Dataset
from transform import *
logger = None

class OneSubDataLoader(Dataset):

    def __init__(self, config, path, set='train'):
        self.multimodal = config['dataset']['multimodal']
        self.seq_len = config['dataset']['seq_len']
        self.target_idx = config['dataset']['target_idx']
        self.training_mode = config['training_params']['mode']
        self.set = set
        self.path = path
        self.eeg_sr = 100 
        self.hbo_sr = 25
        self.hb_sr = 25
        self.ppg_sr = 25
        self.eog_sr = 100

        self.eeg, self.hbo, self.hb, self.ppg, self.eog, self.labels, self.score, self.epochs = self.load_dataset(self.path)
        
        # self.fnirs_transform = Compose(transforms=[ChannelShuffle(),
        #                                            FnirsRandomAmplitudeScale()])

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        eeg_n_sample = 30 * self.eeg_sr * self.seq_len
        hbo_n_sample = 30 * self.hbo_sr * self.seq_len
        hb_n_sample = 30 * self.hb_sr * self.seq_len
        ppg_n_sample = 30 * self.ppg_sr * self.seq_len
        eog_n_sample = 30 * self.eog_sr * self.seq_len
        score_n_sample = 6 * self.seq_len
        idx, seq_len = self.epochs[idx]
        eegs = self.eeg[0][idx:idx+seq_len]
        
        hbos, hbs, ppgs, eogs = None, None, None, None
        if self.hbo:
            hbos = self.hbo[0][idx:idx+seq_len]
        if self.hb:
            hbs = self.hb[0][idx:idx+seq_len]
        if self.ppg:
            ppgs = self.ppg[0][idx:idx+seq_len]
        if self.eog:
            eogs = self.eog[0][idx:idx+seq_len]

        if self.set == 'train':
            
            if self.training_mode in ['scratch', 'fullyfinetune', 'freezefinetune']:
                
                eegs = eegs.reshape(-1, eeg_n_sample)
                eegs = torch.from_numpy(eegs).float()
                if hbos is not None:
                    hbos = hbos.reshape(-1, hbo_n_sample)
                    # hbos = self.fnirs_transform(hbos)
                    hbos = torch.from_numpy(hbos).float()
                else:
                    hbos = [0]
                if hbs is not None:
                    hbs = hbs.reshape(-1, hb_n_sample)
                    # hbs = self.fnirs_transform(hbs)
                    hbs = torch.from_numpy(hbs).float()
                else:
                    hbs = [0]
                if ppgs is not None:
                    ppgs = ppgs.reshape(-1, ppg_n_sample)
                    # ppgs = self.fnirs_transform(ppgs)
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
                hbos = torch.from_numpy(hbos).float()
            else:
                hbos = [0]
            if hbs is not None:
                hbs = hbs.reshape(-1, hb_n_sample)
                hbs = torch.from_numpy(hbs).float()
            else:
                hbs = [0]
            if ppgs is not None:
                ppgs = ppgs.reshape(-1, ppg_n_sample)
                ppgs = torch.from_numpy(ppgs).float()
            else:
                ppgs = [0]
                
            if eogs is not None:
                eogs = eogs.reshape(-1, eog_n_sample)
                eogs = torch.from_numpy(eogs).float()
                eegs = torch.cat((eegs, eogs), dim=0)

        labels = self.labels[0][idx:idx+seq_len]
        labels = torch.from_numpy(labels).long()
        labels = labels[self.target_idx]

        return eegs, hbos, hbs, ppgs, labels

    def load_dataset(self, path):

        eeg, hbo, hb, ppg, eog, labels, epochs = [], [], [], [], [], [], []
        score = []
            
        npz_file = np.load(path)
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
            epochs.append([i, seq_len])
        if not hbo:
            hbo = None
        if not hb:
            hb = None
        if not ppg:
            ppg = None
        if not eog:
            eog = None

        return eeg, hbo, hb, ppg, eog, labels, score, epochs

logger = None

def cleanup():
    cur_rank = dist.get_rank()
    print('rank {}, Cleaning up'.format(cur_rank))
    torch.distributed.barrier()
    print('rank {}, Barrier passed'.format(cur_rank))
    torch.distributed.destroy_process_group()
    print('rank {}, Process group destroyed'.format(cur_rank))
    gc.collect()
    torch.cuda.empty_cache()
    print('rank {}, Cleaned up complete'.format(cur_rank))


class OneFoldTrainer:
    def __init__(self, args, fold, subject_train_path, subject_test_path, config):
        self.args = args
        self.fold = fold
        self.subject_train_path = subject_train_path
        self.subject_test_path = subject_test_path
        self.multimodal = [True if args.eeg else False, True if args.hbo else False, True if args.hb else False, True if args.ppg else False, True if args.eog else False]
        if not (True in self.multimodal):
            raise ValueError('At least one modality should be True')
        
        self.cfg = config
        self.ds_cfg = config['dataset']
        self.cfg['dataset']['multimodal'] = self.multimodal
        self.fp_cfg = config['feature_pyramid']
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']
        
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)
        suffix = '_eeg' if self.multimodal[0] else ''
        suffix += '_hbo' if self.multimodal[1] else ''
        suffix += '_hb' if self.multimodal[2] else ''
        suffix += '_ppg' if self.multimodal[3] else ''
        suffix += '_eog' if self.multimodal[4] else ''
        if self.local_rank == 0:
            logger.info('[INFO] Config name: {}'.format(config['name'] + suffix))

        self.train_iter = 0
        self.train_loss = 0
        self.start_counter_epochs = 0
        self.model = self.build_model()
        self.dset_cfg = config['dataset']
        self.root_dir = self.dset_cfg['root_dir']
        self.dset_name = self.dset_cfg['name']
        self.eeg_channel = self.dset_cfg['eeg_channel']
        self.seq_len = self.dset_cfg['seq_len']
        self.target_idx = self.dset_cfg['target_idx']

        self.dataset_path = os.path.join(self.root_dir, 'dset', self.dset_name, 'npz', self.eeg_channel)
        self.subject_train_path = os.path.join(self.dataset_path, self.subject_train_path)
        self.subject_test_path = os.path.join(self.dataset_path, self.subject_test_path)
        self.loader_dict = self.build_dataloader()

        self.save_Train_Loss = []
        self.save_Val_Loss = []
        if self.local_rank == 0:
            self.save_Loss_path = os.path.join('results', 'mtcl', config['name'], str(datetime.datetime.now())+'_'+str(fold)) 
            if not os.path.exists(os.path.join(self.save_Loss_path, 'train')):
                os.makedirs(os.path.join(self.save_Loss_path, 'train'))
            if not os.path.exists(os.path.join(self.save_Loss_path, 'val')):
                os.makedirs(os.path.join(self.save_Loss_path, 'val'))
        
        self.criterion = nn.CrossEntropyLoss()
        self.activate_train_mode()

        target_lr = self.tp_cfg['target_lr']
        self.optimizer = optim.AdamW(self.model.parameters(), lr=target_lr, weight_decay=self.tp_cfg['weight_decay'])

        self.ckpt_path = os.path.join('checkpoints', 'FnirsFinetune', config['name'] + suffix)
        self.ckpt_name = 'ckpt_{}-to-{}.pth'.format(self.subject_train_path.split('/')[-1].removesuffix('.npz'), 
                                                    self.subject_test_path.split('/')[-1].removesuffix('.npz'))


    def build_model(self):
        model = MainModel(self.cfg)
        model = model.to(self.device)
        if self.local_rank == 0:
            logger.info('[INFO] Number of params of model: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        
        if self.tp_cfg['mode'] != 'scratch':
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            if self.local_rank == 0:
                logger.info('[INFO] Model loaded for finetune')
            load_name = self.cfg['name']
            # load_name = load_name.replace('SL-{:02d}'.format(self.ds_cfg['seq_len']), 'SL-01')
            # load_name = load_name.replace('numScales-{}'.format(self.fp_cfg['num_scales']), 'numScales-1')
            load_name = load_name.replace('private', 'freezefinetune')
            modaltext = '_eeg' if self.multimodal[0] else ''
            modaltext += '_hbo' if self.multimodal[1] else ''
            modaltext += '_hb' if self.multimodal[2] else ''
            modaltext += '_ppg' if self.multimodal[3] else ''
            modaltext += '_eog' if self.multimodal[4] else ''
            load_name = load_name + modaltext
            load_path = os.path.join('checkpoints', load_name, 'ckpt_fold-{0:02d}.pth'.format(self.fold))
            if self.local_rank == 0:
                logger.info('[INFO] Loading model from: {}'.format(load_path))
            model.load_state_dict(torch.load(load_path, weights_only=False), strict=False)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)
        if self.local_rank == 0:
            logger.info('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model
    
    def build_dataloader(self):

        train_dataset = OneSubDataLoader(self.cfg, self.subject_train_path, set='train')
        train_sampler = DistributedSampler(train_dataset, drop_last=False)
        train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

        test_dataset = OneSubDataLoader(self.cfg, self.subject_test_path, set='test')
        test_sampler = DistributedSampler(test_dataset)
        test_loader = DataLoader(dataset=test_dataset, sampler=test_sampler, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
        if self.local_rank == 0:
            logger.info('[INFO] Dataloader prepared, rank: {}'.format(self.local_rank))

        return {'train':train_loader, 'test': test_loader}
    
    def activate_train_mode(self):
        self.model.train()
        # if self.tp_cfg['mode'] == 'freezefinetune':
        #     if self.local_rank == 0:
        #         logger.info('[INFO] Freeze backbone')
        #     self.model.module.encoder.eegEncoder.train(False)
        #     for p in self.model.module.encoder.eegEncoder.parameters():
        #         p.requires_grad = False



    def train_one_epoch(self, epoch):
        correct, total = 0, 0
        total_correct = 0
        total_epochs = 0
        total_loss = 0
        length_load = 0
        
        self.loader_dict['train'].sampler.set_epoch(epoch)
        params_to_clip = [p for p in self.model.parameters() if p.requires_grad]

        for i, (eeg, hbo, hb, ppg, labels) in enumerate(self.loader_dict['train']):
            loss = 0
            total += labels.size(0)
            # inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)
            # if labels.size(0) < self.tp_cfg['batch_size']:
            #     continue
            eeg = eeg.to(self.device)
            if self.multimodal[1]:
                hbo = hbo.to(self.device)
            else:
                hbo = None
            if self.multimodal[2]:
                hb = hb.to(self.device)
            else:
                hb = None
            if self.multimodal[3]:
                ppg = ppg.to(self.device)
            else:
                ppg = None

            outputs = self.model(eeg=eeg, hbo=hbo, hb=hb, ppg=ppg)

            outputs_sum = torch.zeros_like(outputs[0])
            for j in range(len(outputs)):
                outputs_sum += outputs[j]
                loss += self.criterion(outputs[j], labels)
                
            self.optimizer.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
            self.optimizer.step()
            # for name, param in self.model.named_parameters():
            #     if param.grad is None:
            #         print(f"Parameter {name} is not used in the computation graph.")

            self.train_loss += loss.item()
            total_loss = loss.to(self.local_rank)
            dist.reduce(total_loss, dst=0)
            
            predicted = torch.argmax(outputs_sum, 1)
            correct += predicted.eq(labels).sum().item()
            
            total_correct = torch.tensor(correct).to(self.local_rank)
            dist.reduce(total_correct, dst=0)
            total_epochs = torch.tensor(total).to(self.local_rank)
            dist.reduce(total_epochs, dst=0)
            dist.barrier()
            length_load += 1
            
            self.train_iter += 1
            
            if self.local_rank == 0:
                progress_bar(i, len(self.loader_dict['train']), 'Loss: %.3f | Acc: %.3f%% | lr: %.6f'
                        % (total_loss.item() / dist.get_world_size(), 100. * total_correct / total_epochs, get_lr(self.optimizer)))
            
        dist.barrier()
        epoch_loss = torch.tensor(self.train_loss).to(self.local_rank)
        dist.reduce(epoch_loss, dst=0)
        if self.local_rank == 0:
            print()
            logger.info("Epoch average loss: {}".format(epoch_loss.item() / dist.get_world_size() / length_load))
            self.save_Train_Loss.append(epoch_loss.item() / dist.get_world_size() / length_load)
        self.train_loss = 0
        # val_acc, val_loss = self.evaluate(mode='val')
        # dist.barrier()
        # if self.local_rank == 0:
        #     self.early_stopping(val_acc, val_loss, self.model)

        self.activate_train_mode()
            
    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        total_correct = 0
        total_epochs = 0
        total_loss = 0
        epoch_loss = 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.cfg['classifier']['num_classes']))

        for i, (eeg, hbo, hb, ppg, labels) in enumerate(self.loader_dict[mode]):
            loss = 0
            total += labels.size(0)
            # inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)
            
            eeg = eeg.to(self.device)
            if self.multimodal[1]:
                hbo = hbo.to(self.device)
            else:
                hbo = None
            if self.multimodal[2]:
                hb = hb.to(self.device)
            else:
                hb = None
            if self.multimodal[3]:
                ppg = ppg.to(self.device)
            else:
                ppg = None

            outputs = self.model(eeg=eeg, hbo=hbo, hb=hb, ppg=ppg)

            outputs_sum = torch.zeros_like(outputs[0])
            for j in range(len(outputs)):
                outputs_sum += outputs[j]
                loss += self.criterion(outputs[j], labels)
            # outputs_sum = outputs[2]
            # loss = self.criterion(outputs_sum, labels)
            eval_loss += loss.item()
            total_loss = loss.to(self.local_rank)
            dist.reduce(total_loss, dst=0)
                
            predicted = torch.argmax(outputs_sum, 1)
            correct += predicted.eq(labels).sum().item()
            total_correct = torch.tensor(correct).to(self.local_rank)
            dist.reduce(total_correct, dst=0)
            total_epochs = torch.tensor(total).to(self.local_rank)
            dist.reduce(total_epochs, dst=0)
            dist.barrier()
            
            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs_sum.cpu().numpy()])
            
            if self.local_rank == 0:
                progress_bar(i, len(self.loader_dict[mode]), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss / dist.get_world_size(), 100. * total_correct / total_epochs, total_correct, total_epochs))

        epoch_loss = torch.tensor(eval_loss).to(self.local_rank)
        dist.reduce(epoch_loss, dst=0)
        dist.barrier()
        if self.local_rank == 0 and mode == 'val':
            self.save_Val_Loss.append(epoch_loss.item() / dist.get_world_size() / len(self.loader_dict[mode]))
            logger.info("Epoch loss: %.3f, average: %.3f, ACC: %.3f%%"%(epoch_loss.item(), epoch_loss.item() / dist.get_world_size() / len(self.loader_dict[mode]), 100. * total_correct / total_epochs))
        if self.local_rank == 0 and mode == 'test':
            print("Epoch loss: %.3f, average: %.3f, ACC: %.3f%%"%(epoch_loss.item(), epoch_loss.item() / dist.get_world_size() / len(self.loader_dict[mode]), 100. * total_correct / total_epochs))
        if mode == 'val':
            if self.local_rank == 0:
                return 100. * total_correct / total_epochs, epoch_loss
            return None, None
        elif mode == 'test':
            return y_true, y_pred
        else:
            raise NotImplementedError
    
    def run(self):

        for epoch in range(20):
            if self.local_rank == 0:
                print()
                logger.info('[INFO] Fold: {}, Epoch: {}, Subject: {}'.format(self.fold, epoch, 
                                                                             self.subject_train_path.split('/')[-1].removesuffix('.npz')))

            self.train_one_epoch(epoch)
        if self.local_rank == 0:
            np.save(os.path.join(self.save_Loss_path, 'train', 'train'), self.save_Train_Loss) 
            np.save(os.path.join(self.save_Loss_path, 'val', 'val'), self.save_Val_Loss)
        
        if self.local_rank == 0:
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, self.ckpt_name))
        y_true, y_pred = self.evaluate(mode='test')
        print('')

        return y_true, y_pred

def main(args):
    global logger
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    torch.distributed.init_process_group(backend="nccl")
    if dist.get_rank() == 0:
        logger = Logger('results/log/{}.log'.format(str(datetime.datetime.now()).replace(':','-')))
    
    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    for fold in range(1, config['dataset']['num_splits'] + 1):
        split_idx_list = np.load(os.path.join('./split_idx', 'idx_{}.npy'.format(config['dataset']['name'])), allow_pickle=True)
        for subject in split_idx_list[fold-1]['test']:
            p1 = 'S{:02d}D1.npz'.format(subject)
            p2 = 'S{:02d}D2.npz'.format(subject)
            for subject_train_path, subject_test_path in [(p1, p2), (p2, p1)]:  # two directions

                trainer = OneFoldTrainer(args, fold, subject_train_path, subject_test_path, config)
                y_true, y_pred = trainer.run()
                y_true = torch.Tensor(y_true).to(trainer.device)
                y_pred = torch.Tensor(y_pred).to(trainer.device)

                gather_y_true = None
                gather_y_pred = None
                if trainer.local_rank == 0:
                    gather_y_true = [torch.zeros_like(y_true) for _ in range(dist.get_world_size())]
                    gather_y_pred = [torch.zeros_like(y_pred) for _ in range(dist.get_world_size())]
                else:
                    gather_y_true = None
                    gather_y_pred = None
                dist.barrier()
                    
                dist.gather(y_pred, gather_list=gather_y_pred, dst=0)
                dist.gather(y_true, gather_list=gather_y_true, dst=0)
                if trainer.local_rank == 0:
                    Y_true = np.concatenate([Y_true, np.concatenate([y.cpu().numpy() for y in gather_y_true])])
                    Y_pred = np.concatenate([Y_pred, np.concatenate([y.cpu().numpy() for y in gather_y_pred])])
            
                    summarize_result(config, fold, Y_true, Y_pred)
    
    cleanup()
    
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=66, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--eeg', type=int, default=True, help='config file path')
    parser.add_argument('--hbo', type=int, default=True, help='determine hbo modal input')
    parser.add_argument('--hb', type=int, default=True, help='determine hb modal input')
    parser.add_argument('--eog', type=int, default=True, help='determine eog modal input')
    parser.add_argument('--ppg', type=int, default=True, help='determine ppg modal input')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)

