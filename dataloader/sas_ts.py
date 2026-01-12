from .base import AbstractDataloader

import torch
import random
import numpy as np
import torch.utils.data as data_utils


def worker_init_fn(worker_id):
    random.seed(np.random.get_state()[1][0] + worker_id)
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class SASDataloader_ts(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.args = args
        self.rng = np.random
        self.save_folder = dataset._get_preprocessed_folder_path()  # data/preprocessed/...
        dataset = dataset.load_dataset_ts()  # 处理成pkl文件，加载preprocessed中的pkl文件
        # dataset返回划分好的序列数据，训练集、验证集(倒数第二交互)、测试集(倒一)、uid映射、sid映射
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        # 时间数据
        self.train_ts = dataset['train_ts']
        self.val_ts = dataset['val_ts']
        self.test_ts = dataset['test_ts']

        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        args.num_users = self.user_count
        args.num_items = self.item_count
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.sliding_size = args.sliding_window_size  # 1
        self.CLOZE_MASK_TOKEN = self.item_count + 1

    @classmethod
    def code(cls):
        return 'sas'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=torch.cuda.is_available(),
                                           num_workers=self.args.num_workers,
                                           worker_init_fn=worker_init_fn)
        return dataloader

    def _get_train_dataset(self):
        dataset = SASTrainDataset_ts(
            self.args, self.train, self.train_ts, self.max_len, self.sliding_size, self.rng
        )
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                           pin_memory=torch.cuda.is_available(), num_workers=self.args.num_workers)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = SASValidDataset_ts(self.args, self.train, self.val, self.val_ts, self.max_len, self.rng)
        elif mode == 'test':
            dataset = SASTestDataset_ts(self.args, self.train, self.val, self.test, self.test_ts, self.max_len, self.rng)
        return dataset


class SASTrainDataset_ts(data_utils.Dataset):
    def __init__(self, args, u2seq, u2ts, max_len, sliding_size, rng):
        self.args = args
        self.max_len = max_len
        self.sliding_step = int(sliding_size * max_len)  # 1 * max_len
        self.num_items = args.num_items
        self.rng = rng
        self.u2ts = u2ts

        self.all_seqs = []
        self.all_tss = []  # 存储与序列对应的时间戳序列
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            ts_seq = u2ts[u]  # 获取该用户的时间戳序列
            assert len(seq) == len(ts_seq), "序列长度与时间戳长度不匹配"

            if len(seq) < self.max_len + self.sliding_step:
                self.all_seqs.append(seq)
                self.all_tss.append(ts_seq)
            else:   # 如果序列很长，就从末尾开始提取 max_len长度的子序列
                # 例如序列长度100，max_len=40，slid_step=40，start_idx=[60, 20]。会得到[60:100], [20:60]两个子序列
                start_idx = range(len(seq) - max_len, -1, -self.sliding_step)
                self.all_seqs.extend([seq[i:i + max_len] for i in start_idx])       # 子序列都加入all_seqs
                self.all_tss.extend([ts_seq[i:i + max_len] for i in start_idx])     # 对应时间戳子序列

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        seq = self.all_seqs[index]
        ts_seq = self.all_tss[index]  # 获取对应时间戳序列

        labels = seq[-self.max_len:]
        tokens = seq[:-1][-self.max_len:]

        mask_len = self.max_len - len(tokens)  # 3 - 2 = 1
        tokens = [0] * mask_len + tokens  # [0, 6, 7]

        mask_len = self.max_len - len(labels)  # 0
        labels = [0] * mask_len + labels  # [6, 7, 8]     6对应7，7对应8，next_token预测

        # 时间间隔计算：相邻时间戳的差（长度为len(ts_seq)-1）
        if len(ts_seq) <= 1:
            timespans = [1.0] * len(tokens)  # 若序列过短，默认间隔为1
        else:
            ts_diff = np.diff(ts_seq)  # 计算相邻时间差：t2-t1, t3-t2, ...
            ts_diff = ts_diff[-self.max_len:]  # 截取与tokens等长的部分
            # 不足max_len时前面补1
            ts_mask_len = self.max_len - len(ts_diff)
            timespans = np.pad(ts_diff, (ts_mask_len, 0), mode='constant', constant_values=1.0).tolist()
            timespans = [np.log1p(ts) for ts in timespans]  # log(1+ts)，避免0值
        return (
            torch.LongTensor(tokens),
            torch.LongTensor(labels),
            torch.FloatTensor(timespans)  # 时间间隔
        )


class SASValidDataset_ts(data_utils.Dataset):
    def __init__(self, args, u2seq, u2answer, u2ts, max_len, rng):
        self.args = args
        self.u2seq = u2seq
        self.u2answer = u2answer
        self.u2ts = u2ts  # 验证集时间戳
        users = sorted(self.u2seq.keys())
        self.users = [u for u in users if len(u2answer[u]) > 0]
        self.max_len = max_len
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        ts_seq = self.u2ts[user]  # 用户的时间戳序列
        answer = self.u2answer[user]

        # 物品序列处理
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        # 时间间隔计算
        if len(ts_seq) <= 1:
            timespans = [1.0] * self.max_len
        else:
            ts_diff = np.diff(ts_seq)
            ts_diff = ts_diff[-self.max_len:]
            ts_mask_len = self.max_len - len(ts_diff)
            timespans = np.pad(ts_diff, (ts_mask_len, 0), mode='constant', constant_values=1.0).tolist()
            timespans = [np.log1p(ts) for ts in timespans]  # log(1+ts)，避免0值
        return (
            torch.LongTensor(seq),
            torch.LongTensor(answer),
            torch.FloatTensor(timespans)
        )


class SASTestDataset_ts(data_utils.Dataset):
    def __init__(self, args, u2seq, u2val, u2answer, u2ts, max_len, rng):
        self.args = args
        self.u2seq = u2seq
        self.u2val = u2val
        self.u2answer = u2answer
        self.u2ts = u2ts
        users = sorted(self.u2seq.keys())
        self.users = [u for u in users if len(u2val[u]) > 0 and len(u2answer[u]) > 0]
        self.max_len = max_len
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2val[user]
        ts_seq = self.u2ts[user]
        answer = self.u2answer[user]

        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        # 时间间隔计算
        if len(ts_seq) <= 1:
            timespans = [1.0] * self.max_len
        else:
            ts_diff = np.diff(ts_seq)
            ts_diff = ts_diff[-self.max_len:]
            ts_mask_len = self.max_len - len(ts_diff)
            timespans = np.pad(ts_diff, (ts_mask_len, 0), mode='constant', constant_values=1.0).tolist()
            timespans = [np.log1p(ts) for ts in timespans]  # log(1+ts)，避免0值
        return (
            torch.LongTensor(seq),
            torch.LongTensor(answer),
            torch.FloatTensor(timespans)  # 新增时间间隔
        )

