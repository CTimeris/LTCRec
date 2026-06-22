import torch
import numpy as np
import argparse
import random

RAW_DATASET_ROOT_FOLDER = 'data'
EXPERIMENT_ROOT = 'experiments'

STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'

PROJECT_NAME = 'ltcrec'


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def set_template(args):
    args.min_uc = 5
    args.min_sc = 5
    args.split = 'leave_one_out'
    
    if args.dataset_code == None:
        print('******************** Dataset Selection ********************')
        dataset_code = {'1': 'ml-1m', 'b': 'beauty', 's': 'sports', 't': 'steam', 'v': 'video', 'x': 'xlong'}
        args.dataset_code = dataset_code[input('Input 1 for ml-1m, b for beauty, s for sports, t for steam, v for video and x for xlong: ')]
    
    if args.dataset_code == 'ml-1m':
        args.max_len = 200
        args.val_iterations = 500
    elif args.dataset_code == 'steam':
        args.max_len = 50
        args.val_iterations = 2000
    elif args.dataset_code == 'xlong':
        args.max_len = 1000
        args.val_iterations = 2000
    else:  # beauty, sports, video, yelp, ml-100k
        args.max_len = 50
        args.val_iterations = 1000
    
    batch = 32 if args.dataset_code == 'xlong' else 128
    args.train_batch_size = batch
    args.val_batch_size = batch * 2
    args.test_batch_size = batch * 2

    if torch.cuda.is_available(): args.device = 'cuda'
    else: args.device = 'cpu'
    args.optimizer = 'AdamW'
    if args.lr is None: args.lr = 0.001
    if args.weight_decay is None: args.weight_decay = 0.01
    if args.dropout is None: args.dropout = 0.2
    if args.attn_dropout is None: args.attn_dropout = 0.2
    if args.mask_prob is None: args.mask_prob = 0.2
    
    args.enable_lr_schedule = False
    args.decay_step = 10000
    args.gamma = 0.1
    args.enable_lr_warmup = False
    args.warmup_steps = 100

    args.metric_ks = [1, 5, 10, 20, 50]
    args.best_metric = 'Recall@10'
    args.num_blocks = 2
    args.num_heads = 2
    args.head_size = None


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='ml-1m')
parser.add_argument('--min_rating', type=int, default=0)
parser.add_argument('--min_uc', type=int, default=2)
parser.add_argument('--min_sc', type=int, default=1)
parser.add_argument('--split', type=str, default='leave_one_out')
parser.add_argument('--seed', type=int, default=42)

################
# Dataloader
################
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--sliding_window_size', type=float, default=1)
parser.add_argument('--negative_sample_size', type=int, default=100)
parser.add_argument('--xlong_negative_sample_size', type=int, default=10000)
parser.add_argument('--time_transform', type=str, default='none', choices=['none', 'log1p'])
parser.add_argument('--time_scale', type=float, default=1.0)
parser.add_argument('--max_time_span', type=float, default=0.0)

################
# Trainer
################
# device #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
# optimizer & lr#
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam'])
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_grad_norm', type=float, default=5.0)
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--decay_step', type=int, default=10000)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--enable_lr_warmup', type=bool, default=True)
parser.add_argument('--warmup_steps', type=int, default=50)

# evaluation #
parser.add_argument('--val_strategy', type=str, default='iteration', choices=['epoch', 'iteration'])
parser.add_argument('--val_iterations', type=int, default=50)  # only for iteration val_strategy
parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--early_stopping_patience', type=int, default=3)
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 20, 50])
parser.add_argument('--best_metric', type=str, default='Recall@10')
parser.add_argument('--use_wandb', type=str2bool, nargs='?', const=True, default=False)

################
# Model
################
parser.add_argument('--model_code', type=str, default='ltcrec', choices=['ltcrec', 'ltcrec_ode'])
parser.add_argument('--max_len', type=int, default=None)       # 序列长度
parser.add_argument('--hidden_units', type=int, default=64)
parser.add_argument('--num_blocks', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--head_size', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--attn_dropout', type=float, default=0.3)
parser.add_argument('--mask_prob', type=float, default=None)
parser.add_argument('--ltc_epsilon', type=float, default=1e-6)


args = parser.parse_args()
