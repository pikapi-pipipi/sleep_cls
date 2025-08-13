import os
import json
import argparse
import warnings
import gc
import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.nn.functional import all_gather

from utils import *
from loss import SupConLoss
from loader import EEGDataLoader
from models.main_model import MainModel

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
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        self.multimodal = [True if args.eeg else False, True if args.hbo else False, True if args.hb else False, True if args.ppg else False, True if args.eog else False]
        if not (True in self.multimodal):
            raise ValueError('At least one modality should be True')

        self.cfg = config
        self.tp_cfg = config['training_params']
        self.cfg['dataset']['multimodal'] = self.multimodal
        self.es_cfg = self.tp_cfg['early_stopping']
        
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)
        suffix = '_eeg' if self.multimodal[0] else ''
        suffix += '_hbo' if self.multimodal[1] else ''
        suffix += '_hb' if self.multimodal[2] else ''
        suffix += '_ppg' if self.multimodal[3] else ''
        suffix += '_eog' if self.multimodal[4] else ''
        if self.local_rank == 0:
            logger.info('[INFO] Config name: {}'.format(config['name']+suffix))
        
        self.train_iter = 0
        self.train_loss = 0
        self.start_counter_epochs = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        
        self.save_Train_Loss = []
        self.save_Val_Loss = []
        if self.local_rank == 0:
            self.save_Loss_path = os.path.join('results', 'crl', config['name'], str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))+'_'+str(fold)) 
            if not os.path.exists(os.path.join(self.save_Loss_path, 'train')):
                os.makedirs(os.path.join(self.save_Loss_path, 'train'))
            if not os.path.exists(os.path.join(self.save_Loss_path, 'val')):
                os.makedirs(os.path.join(self.save_Loss_path, 'val'))
                
        self.criterion = SupConLoss(temperature=self.tp_cfg['temperature'])

        target_lr = self.tp_cfg['target_lr']
        self.optimizer = optim.AdamW(self.model.parameters(), lr=target_lr, weight_decay=self.tp_cfg['weight_decay'])
        
        self.ckpt_path = os.path.join('checkpoints', config['name'] + suffix)
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path, ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])

    def build_model(self):
        model = MainModel(self.cfg)
        model = model.to(self.device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if self.local_rank == 0:
            logger.info('[INFO] Number of params of model: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        model = torch.nn.parallel.DistributedDataParallel(model)
        if self.local_rank == 0:
            logger.info('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.local_rank))

        return model
    
    def build_dataloader(self):
        dataloader_args = {'batch_size': self.tp_cfg['batch_size'], 'num_workers': 8, 'pin_memory': True}
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, **dataloader_args)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = DataLoader(dataset=val_dataset, sampler=val_sampler, **dataloader_args)
        if self.local_rank == 0:
            logger.info('[INFO] Dataloader prepared, rank: {}'.format(self.local_rank))

        return {'train': train_loader, 'val': val_loader}

    def train_one_epoch(self, epoch):
        self.model.train()
        length_load = 0
        self.loader_dict['train'].sampler.set_epoch(epoch)

        for i, (eeg, hbo, hb, ppg, eog, labels) in enumerate(self.loader_dict['train']):
            loss = 0
            loss1 = 0
            labels = labels.view(-1).to(self.device)
            if labels.size(0) < self.tp_cfg['batch_size']:
                # skip the last batch if the size is smaller than batch size
                continue
            length_load += 1
            world_size = dist.get_world_size()
            all_labels = torch.cat(all_gather(labels), dim=0)
            # inputs = torch.cat([inputs[0], inputs[1]], dim=0).to(self.device)
            eeg = torch.cat([eeg[0], eeg[1]], dim=0).to(self.device)
            if self.multimodal[1]:
                hbo = torch.cat([hbo, hbo], dim=0).to(self.device)
            else:
                hbo = None
            if self.multimodal[2]:
                hb = torch.cat([hb, hb], dim=0).to(self.device)
            else:
                hb = None
            if self.multimodal[3]:
                ppg = torch.cat([ppg, ppg], dim=0).to(self.device)
            else:    
                ppg = None
            if self.multimodal[4]:
                eog = torch.cat([eog, eog], dim=0).to(self.device)
            else:
                eog = None

            outputs = self.model(eeg=eeg, hbo=hbo, hb=hb, ppg=ppg, eog=eog)

            for j in range(len(outputs)):
                f1, f2 = torch.split(outputs[j], [labels.size(0), labels.size(0)], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                all_features = torch.cat(all_gather(features), dim=0)
                loss1 += self.criterion(all_features, all_labels)
            loss += loss1
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            dist.barrier()

            self.train_loss += loss.item()
            self.train_iter += 1
            if self.local_rank == 0:
                progress_bar(i, len(self.loader_dict['train']), 'Lr:%.4e|Loss:%.3f'%(get_lr(self.optimizer), loss.item()))
            if i == 20:
                break
            

        dist.barrier()
        if self.local_rank == 0:
            print('')
            logger.info('Epoch average Loss: %.3f' %( self.train_loss / length_load))
            self.save_Train_Loss.append(self.train_loss / length_load)
        self.train_loss = 0
        val_loss = self.evaluate(mode='val')
        dist.barrier()
        if self.local_rank == 0:
            self.early_stopping(None, val_loss, self.model)
        self.model.train()

    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        eval_loss = 0

        for i, (eeg, hbo, hb, ppg, eog, labels) in enumerate(self.loader_dict[mode]):
            loss = 0
            loss1 = 0
            # inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)
            world_size = dist.get_world_size()
            all_labels = torch.cat(all_gather(labels), dim=0) 
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
            if self.multimodal[4]:
                eog = eog.to(self.device)
            else:
                eog = None
            outputs = self.model(eeg=eeg, hbo=hbo, hb=hb, ppg=ppg, eog=eog)

            for j in range(len(outputs)):
                features = outputs[j]
                all_features = torch.cat(all_gather(features), dim=0).unsqueeze(1).repeat(1, 2, 1)
                loss1 += self.criterion(all_features, all_labels)
            loss += loss1

            eval_loss += loss.item()
            
            if self.local_rank == 0:
                progress_bar(i, len(self.loader_dict[mode]), 'Loss:%.3f|' %(loss.item()))
        
        if self.local_rank == 0:
            logger.info("Epoch loss: %.3f, average: %.3f"%(eval_loss, eval_loss / len(self.loader_dict[mode])))
            # progress_bar(len(self.loader_dict[mode]), len(self.loader_dict[mode]), 'Loss:%.3f|crl:%.3f|prl:%.3f' %(get_lr(self.optimizer), epoch_loss.item() / dist.get_world_size() / len(self.loader_dict[mode])))
        
        if mode == 'val' and self.local_rank == 0:
            self.save_Val_Loss.append(eval_loss / len(self.loader_dict[mode]))
        if self.local_rank == 0:
            return eval_loss
        else:
            return None
    
    def run(self):
        break_flag = 0
        all_break_flag = 0
        for epoch in range(self.tp_cfg['max_epochs']):
            if self.early_stopping.start_Counter == False and epoch >= self.start_counter_epochs:
                self.early_stopping.start_Counter = True
            if self.local_rank == 0:
                print()
                logger.info('[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch(epoch=epoch)
            if self.local_rank == 0:
                np.save(os.path.join(self.save_Loss_path, 'train', 'train'), self.save_Train_Loss) 
                np.save(os.path.join(self.save_Loss_path, 'val', 'val'), self.save_Val_Loss)
                if self.early_stopping.early_stop:
                    logger.info('Rank: {}, Early stopping at epoch: {}'.format(self.local_rank, epoch))
                    break_flag = 1
            all_break_flag = torch.tensor(break_flag).to(self.local_rank)
            dist.all_reduce(all_break_flag)
            dist.barrier()
            if all_break_flag:
                break


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
    
    for fold in range(1, config['dataset']['num_splits'] + 1):
        trainer = OneFoldTrainer(args, fold, config)
        trainer.run()
        # del trainer
    cleanup()
        
        

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=66, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--eeg', type=int, default=True, help='determine eeg modal input')
    parser.add_argument('--hbo', type=int, default=True, help='determine hbo modal input')
    parser.add_argument('--hb', type=int, default=True, help='determine hb modal input')
    parser.add_argument('--eog', type=int, default=True, help='determine eog modal input')
    parser.add_argument('--ppg', type=int, default=True, help='determine ppg modal input')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # main()
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    main(args)
