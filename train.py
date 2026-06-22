import torch
import numpy as np
import random

from config import *
from model import *
from dataloader import *
from trainer import *


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args, export_root=None):
    seed_everything(args.seed)
    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/' + args.model_code + '/' + args.dataset_code + \
            '_' + str(args.weight_decay) + '_' + str(args.dropout) + '_' + str(args.attn_dropout)

    train, val, test = dataloader_factory(args)     # train_dataloader、val_dataloader、test_dataloader
    model = LTCRec(args)
    trainer = LTCRecTrainer(args, model, train, val, test, export_root, args.use_wandb)
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    set_template(args)
    train(args)
