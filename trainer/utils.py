from config import *

import json
import os
import pprint as pp
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim as optim


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                         for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def absolute_recall_mrr_ndcg_for_ks(scores, labels, ks):        # score: b,n
    metrics = {}
    labels = F.one_hot(labels, num_classes=scores.size(1))
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)     # 负分升序 = 原始得分降序排序（大的排前面）, rank得到的是下标

    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        # 假设标签为[0,0,1,0]，表示index=2的位置是正例，cut为推荐列表[3,2,0]，表示推荐的索引，那么hits为[0,1,0]，第二个命中
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()
        
        metrics['MRR@%d' % k] = \
            (hits / torch.arange(1, k+1).unsqueeze(0).to(labels.device)).sum(1).mean().cpu().item()     # 命中位置的倒数

        position = torch.arange(2, 2+k)                     # 2, 3, ... 2+k
        weights = 1 / torch.log2(position.float())          # 1/log2, 1/log3, ..., 1/log(2+k)，越前面越重要
        dcg = (hits * weights.to(hits.device)).sum(1)       # 命中位置会保留权重
        idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)  # 最大dcg
        ndcg = (dcg / idcg).mean()                  # ndcg = dcg / idcg
        metrics['NDCG@%d' % k] = ndcg.cpu().item()

        metrics['HitRate@%d' % k] = (hits.sum(1) > 0).float().mean().cpu().item()
    return metrics


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
