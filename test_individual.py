import os
import json
import argparse
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from train_individual import OneFoldTrainer, OneSubDataLoader
from models.main_model import MainModel
import torch.distributed as dist
import gc
from torch.utils.data.distributed import DistributedSampler


def clean_up():
    cur_rank = dist.get_rank()
    print('rank {}, Cleaning up'.format(cur_rank))
    torch.distributed.barrier()
    print('rank {}, Barrier passed'.format(cur_rank))
    torch.distributed.destroy_process_group()
    print('rank {}, Process group destroyed'.format(cur_rank))
    gc.collect()
    torch.cuda.empty_cache()
    print('rank {}, Cleaned up complete'.format(cur_rank))
    
class OneFoldEvaluator(OneFoldTrainer):
    def __init__(self, args, fold, subject_train_path, subject_test_path, config):
        self.args = args
        self.fold = fold
        self.subject_train_path = subject_train_path
        self.subject_test_path = subject_test_path
        self.multimodal = [True if args.eeg else False, True if args.hbo else False, True if args.hb else False, True if args.ppg else False, True if args.eog else False]
        if not (True in self.multimodal):
            raise ValueError('At least one modality should be True')
        
        self.cfg = config
        self.cfg['dataset']['multimodal'] = self.multimodal
        self.ds_cfg = config['dataset']
        self.tp_cfg = config['training_params']
        
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)
        suffix = '_eeg' if self.multimodal[0] else ''
        suffix += '_hbo' if self.multimodal[1] else ''
        suffix += '_hb' if self.multimodal[2] else ''
        suffix += '_ppg' if self.multimodal[3] else ''
        suffix += '_eog' if self.multimodal[4] else ''
        self.suffix = suffix
        if self.local_rank == 0:
            print('[INFO] Config name: {}'.format(config['name']))
            
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
        
        self.criterion = nn.CrossEntropyLoss()
        self.ckpt_path = os.path.join('checkpoints', 'FnirsFinetune', config['name'] + suffix)
        self.ckpt_name = 'ckpt_{}-to-{}.pth'.format(self.subject_train_path.split('/')[-1].removesuffix('.npz'), 
                                                    self.subject_test_path.split('/')[-1].removesuffix('.npz'))

    def build_model(self):
        model = MainModel(self.cfg)
        model = model.to(self.device)
        if self.local_rank == 0:
            print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
            print(self.cfg['name'] + self.suffix)
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model
    
    def build_dataloader(self):
        test_dataset = OneSubDataLoader(self.cfg, self.subject_test_path, set='test')
        test_sampler = DistributedSampler(test_dataset)
        test_loader = DataLoader(dataset=test_dataset, sampler=test_sampler, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
        print('[INFO] Dataloader prepared， rank: {}'.format(self.local_rank))

        return {'test': test_loader} 
   
    def run(self):
        if self.local_rank == 0:
            print('\n[INFO] Fold: {}, Subject: {}-to-{}'.format(self.fold, 
                                                                self.subject_test_path.split('/')[-1].removesuffix('.npz'),
                                                                self.subject_train_path.split('/')[-1].removesuffix('.npz')))
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name), weights_only=False))
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

    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    for fold in range(1, config['dataset']['num_splits'] + 1):
        split_idx_list = np.load(os.path.join('./split_idx', 'idx_{}.npy'.format(config['dataset']['name'])), allow_pickle=True)
        for subject in split_idx_list[fold-1]['test']:
            p1 = 'S{:02d}D1.npz'.format(subject)
            p2 = 'S{:02d}D2.npz'.format(subject)
            for subject_train_path, subject_test_path in [(p1, p2), (p2, p1)]:  # two directions

                evaluator = OneFoldEvaluator(args, fold, subject_train_path, subject_test_path, config)
                y_true, y_pred = evaluator.run()
                y_true = torch.Tensor(y_true).to(evaluator.device)
                y_pred = torch.Tensor(y_pred).to(evaluator.device)

                gather_y_true = None
                gather_y_pred = None
                if evaluator.local_rank == 0:
                    gather_y_true = [torch.zeros_like(y_true) for _ in range(dist.get_world_size())]
                    gather_y_pred = [torch.zeros_like(y_pred) for _ in range(dist.get_world_size())]
                else:
                    gather_y_true = None
                    gather_y_pred = None
                dist.barrier()
                    
                dist.gather(y_pred, gather_list=gather_y_pred, dst=0)
                dist.gather(y_true, gather_list=gather_y_true, dst=0)
                if evaluator.local_rank == 0:
                    Y_true = np.concatenate([Y_true, np.concatenate([y.cpu().numpy() for y in gather_y_true])])
                    Y_pred = np.concatenate([Y_pred, np.concatenate([y.cpu().numpy() for y in gather_y_pred])])
            
                    summarize_result(config, fold, Y_true, Y_pred)
    
        
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
