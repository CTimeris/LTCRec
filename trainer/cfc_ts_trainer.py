from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from .utils import *
from .loggers import *
from .base import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import numpy as np
from abc import *
from pathlib import Path


class CFCTrainer(BaseTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, use_wandb):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root, use_wandb)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        if args.use_ltc:
            self.optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)    # 液态网络对学习率敏感
        else:
            self.optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def calculate_loss(self, batch):
        seqs, labels, timespans = batch  # 接收时间间隔

        if self.args.dataset_code != 'xlong':
            logits = self.model(seqs, timespans=timespans)[0]  # 传递给model
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = self.ce(logits, labels)
        else:
            logits, labels_ = self.model(seqs, labels=labels, timespans=timespans)  # 传递
            logits = logits.view(-1, logits.size(-1))
            labels_[labels == 0] = 0
            labels_ = labels_.view(-1)
            loss = self.ce(logits, labels_)
        return loss

    def calculate_metrics(self, batch):
        seqs, labels, timespans = batch  # 接收时间间隔

        if self.args.dataset_code != 'xlong':
            scores = self.model(seqs, timespans=timespans)[0][:, -1, :]  # 传递
            B, L = seqs.shape
            for i in range(L):
                scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
            scores[:, 0] = -1e9
        else:
            scores, labels = self.model(seqs, labels=labels, timespans=timespans)  # 传递
            scores = scores[:, -1, :]

        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)
        return metrics