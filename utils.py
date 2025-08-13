import os
import sys
import math
import time
import torch
import random
import numpy as np
import sklearn.metrics as skmet
from terminaltables import SingleTable
from termcolor import colored
import torch.optim as optim
import logging


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 25.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, ckpt_path='./checkpoints', ckpt_name='checkpoint.pth', mode='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.start_Counter = False
        if mode == 'max':
            self.init_metric = 0
        elif mode == 'min':
            self.init_metric = -np.inf
        else:
            raise NotImplementedError
            
        self.delta = delta
        self.ckpt_path = ckpt_path
        self.ckpt_name = ckpt_name if '.pth' in ckpt_name else ckpt_name + '.pth'

        os.makedirs(self.ckpt_path, exist_ok=True)


    def __call__(self, val_acc, val_loss, model):
        
        if self.mode == 'max':
            score = val_acc
            val_metric = val_acc
        elif self.mode == 'min':
            score = -val_loss
            val_metric = val_loss
        else:
            raise NotImplementedError

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            if self.start_Counter:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                print(f'[INFO] Validation loss increased {val_metric:.6f}.  Passed ...\n')
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.mode == 'max':
                print(f'[INFO] Validation accuracy increased ({self.init_metric:.6f} --> {val_metric:.6f}).  Saving model ...\n')
            elif self.mode == 'min':
                print(f'[INFO] Validation loss decreased ({self.init_metric:.6f} --> {val_metric:.6f}).  Saving model ...\n')
            else:
                raise NotImplementedError

        torch.save(model.state_dict(), os.path.join(self.ckpt_path, self.ckpt_name))
        self.init_metric = val_metric


def summarize_result(config, fold, y_true, y_pred, save=True):
    os.makedirs('results', exist_ok=True)
    y_pred_argmax = np.argmax(y_pred, 1)
    result_dict = skmet.classification_report(y_true, y_pred_argmax, digits=3, output_dict=True)
    cm = skmet.confusion_matrix(y_true, y_pred_argmax)
    
    accuracy = round(result_dict['accuracy']*100, 1)
    macro_f1 = round(result_dict['macro avg']['f1-score']*100, 1)
    kappa = round(skmet.cohen_kappa_score(y_true, y_pred_argmax), 3)
    
    wpr = round(result_dict['0.0']['precision']*100, 1)
    wre = round(result_dict['0.0']['recall']*100, 1)
    wf1 = round(result_dict['0.0']['f1-score']*100, 1)
    
    n1pr = round(result_dict['1.0']['precision']*100, 1)
    n1re = round(result_dict['1.0']['recall']*100, 1)
    n1f1 = round(result_dict['1.0']['f1-score']*100, 1)

    n2pr = round(result_dict['2.0']['precision']*100, 1)
    n2re = round(result_dict['2.0']['recall']*100, 1)
    n2f1 = round(result_dict['2.0']['f1-score']*100, 1)
    
    n3pr = round(result_dict['3.0']['precision']*100, 1)
    n3re = round(result_dict['3.0']['recall']*100, 1)
    n3f1 = round(result_dict['3.0']['f1-score']*100, 1)
    
    rpr = round(result_dict['4.0']['precision']*100, 1)
    rre = round(result_dict['4.0']['recall']*100, 1)
    rf1 = round(result_dict['4.0']['f1-score']*100, 1)
    
    overall_data = [
        ['ACC', 'MF1', '\u03BA'],
        [accuracy, macro_f1, kappa],
    ]
    
    perclass_data = [
        [colored('A', 'cyan') + '\\' + colored('P', 'green'), 'W', 'N1', 'N2', 'N3', 'R', 'PR', 'RE', 'F1'],
        ['W', cm[0][0], cm[0][1], cm[0][2], cm[0][3], cm[0][4], wpr, wre, wf1],
        ['N1', cm[1][0], cm[1][1], cm[1][2], cm[1][3], cm[1][4], n1pr, n1re, n1f1],
        ['N2', cm[2][0], cm[2][1], cm[2][2], cm[2][3], cm[2][4], n2pr, n2re, n2f1],
        ['N3', cm[3][0], cm[3][1], cm[3][2], cm[3][3], cm[3][4], n3pr, n3re, n3f1],
        ['R', cm[4][0], cm[4][1], cm[4][2], cm[4][3], cm[4][4], rpr, rre, rf1],
    ]
    
    overall_dt = SingleTable(overall_data, colored('OVERALL RESULT', 'red'))
    perclass_dt = SingleTable(perclass_data, colored('PER-CLASS RESULT', 'red'))
    
    print('\n[INFO] Evaluation result from fold 1 to {}'.format(fold))
    print('\n' + overall_dt.table)
    print('\n' + perclass_dt.table)
    print(colored(' A', 'cyan') + ': Actual Class, ' + colored('P', 'green') + ': Predicted Class' + '\n\n')
    
    if save:
        with open(os.path.join('results', config['name'] + '.txt'), 'w') as f:
            f.write(
                str(fold) + ' ' +
                str(round(result_dict['accuracy']*100, 1)) + ' ' + 
                str(round(result_dict['macro avg']['f1-score']*100, 1)) + ' ' + 
                str(round(kappa, 3)) + ' ' +
                str(round(result_dict['0.0']['f1-score']*100, 1)) + ' ' +
                str(round(result_dict['1.0']['f1-score']*100, 1)) + ' ' +
                str(round(result_dict['2.0']['f1-score']*100, 1)) + ' ' +
                str(round(result_dict['3.0']['f1-score']*100, 1)) + ' ' +
                str(round(result_dict['4.0']['f1-score']*100, 1)) + ' '
            )


def set_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value) # Python hash buildin
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

class WarmupCosineAnnealingStepLR:
    def __init__(self, optimizer, warmup_steps, total_steps, target_lr, min_lr=1e-6, last_step=-1, T_mult=1):
        """
        :param optimizer: The optimizer instance, e.g., SGD, Adam
        :param warmup_steps: Number of steps for warmup phase (learning rate increases from initial to target)
        :param total_steps: Total number of training steps for the first cycle
        :param target_lr: The target learning rate after warmup phase
        :param min_lr: Minimum learning rate, which is the lower bound during the cosine annealing phase
        :param last_step: The last updated step (default -1 means starting from the beginning)
        :param T_mult: Factor to multiply total_steps after each restart (default 1 means constant cycle length)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.target_lr = target_lr
        self.min_lr = min_lr
        self.step_counter = last_step
        self.base_lrs = [param_group['lr'] for param_group in optimizer.param_groups]  # Get the initial learning rate
        self.current_cycle = 0
        self.T_mult = T_mult  # Factor for cycle length adjustment

    def step(self):
        """Update the learning rate after each training step."""
        self.step_counter += 1

        # Calculate the steps in the current cycle
        cycle_total_steps = self.total_steps * (self.T_mult ** self.current_cycle)

        if self.step_counter < self.warmup_steps:
            # Warmup phase: linearly increase learning rate
            lr = [(base_lr + (self.target_lr - base_lr) * (self.step_counter / self.warmup_steps)) for base_lr in self.base_lrs]
        else:
            # Cosine Annealing phase: decay learning rate using a cosine function
            current_step_in_cycle = (self.step_counter - self.warmup_steps) % cycle_total_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step_in_cycle / cycle_total_steps))
            lr = [(self.target_lr - self.min_lr) * cosine_decay + self.min_lr for base_lr in self.base_lrs]

        # Update the learning rate in the optimizer
        for param_group, l in zip(self.optimizer.param_groups, lr):
            param_group['lr'] = l

        # Check if we need to restart the cycle
        if (self.step_counter - self.warmup_steps) // cycle_total_steps > 0:
            self.current_cycle += 1
            self.step_counter = -1

    def get_lr(self):
        """Return the current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def restart(self):
        """Manually trigger a restart for the cosine annealing schedule."""
        self.current_cycle += 1
        self.step_counter = -1
    
def Logger(logfile):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
