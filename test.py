import os
import json
import argparse
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from loader import EEGDataLoader
from train_mtcl import OneFoldTrainer
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
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
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
        if self.local_rank == 0:
            print('[INFO] Config name: {}'.format(config['name']))

        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        
        self.criterion = nn.CrossEntropyLoss()
        self.ckpt_path = os.path.join('checkpoints', config['name']+suffix)
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        
    def build_model(self):
        model = MainModel(self.cfg)
        model = model.to(self.device)
        if self.local_rank == 0:
            print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model
    
    def build_dataloader(self):
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_sampler = DistributedSampler(test_dataset)
        test_loader = DataLoader(dataset=test_dataset, sampler=test_sampler, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
        print('[INFO] Dataloader prepared， rank: {}'.format(self.local_rank))

        return {'test': test_loader} 
   
    def run(self):
        if self.local_rank == 0:
            print('\n[INFO] Fold: {}'.format(self.fold))
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        y_true, y_pred = self.evaluate(mode='test')
        print('')

        return y_true, y_pred

def main(args):
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
        evaluator = OneFoldEvaluator(args, fold, config)
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
        
        dist.gather(y_true, gather_y_true, dst=0)
        dist.gather(y_pred, gather_y_pred, dst=0)
        if evaluator.local_rank == 0:
            Y_true = np.concatenate([Y_true, np.concatenate([y.cpu().numpy() for y in gather_y_true])])
            Y_pred = np.concatenate([Y_pred, np.concatenate([y.cpu().numpy() for y in gather_y_pred])])
        
            summarize_result(config, fold, Y_true, Y_pred)
    clean_up()
        
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
