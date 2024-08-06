import torch
import numpy as np
import random
import os
from os.path import join, exists
import pandas as pd

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_deterministic(seed=42):
    # Set seed for random number generators in torch, numpy, and random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for CUDA to make deterministic
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Other potential environment variables for determinism
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_exp_name(args):
    attributes = ["hidden_dim",'seed']
    name = "_".join([str(args[a]) for a in attributes])
    return name    

def generate_random_seeds(n, seed=42):
    random.seed(seed)  # Seed the random number generator for reproducibility
    seeds = [random.randint(0, 2**32 - 1) for _ in range(n)]
    return seeds

def pretty_print_metrics(data):

    for split, metrics in data.items():
        metrics_str = ', '.join(f"{metric}: {value:.2f}" for metric, value in metrics.items())
        print(f"{split.capitalize()} -> {metrics_str}")


def add_new_metrics(hist, new):
    for k, v in new.items():
        hist[k].append(v)
    return hist



def log_results(args, metrics, split, logdir="results"):
    logdir = join(logdir, args.dataset)
    logdir = join(logdir, args.train_method)
    df = pd.DataFrame.from_dict(metrics)

    csv_file_path = join(logdir, f'{create_exp_name(args)}/{split}.csv')
    if not exists(join(logdir, f'{create_exp_name(args)}')):
        os.makedirs(join(logdir, f'{create_exp_name(args)}'))
    df.to_csv(csv_file_path, index=False)


def save_features(args, features, split, logdir="feats"):
    logdir = join(logdir, args.dataset)
    logdir = join(logdir, args.train_method)
    file_path = join(logdir, f'{create_exp_name(args)}/{split}.pth')
    if not exists(join(logdir, f'{create_exp_name(args)}')):
        os.makedirs(join(logdir, f'{create_exp_name(args)}'))
    torch.save(features, file_path)

