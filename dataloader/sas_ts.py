import random

import numpy as np
import torch
import torch.utils.data as data_utils

from .base import AbstractDataloader


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    random.seed(seed)
    np.random.seed(seed)


def _transform_deltas(deltas, args):
    deltas = np.asarray(deltas, dtype=np.float32)
    deltas = np.maximum(deltas, 0.0)

    scale = max(float(getattr(args, "time_scale", 1.0) or 1.0), 1e-12)
    deltas = deltas / scale

    transform = getattr(args, "time_transform", "none")
    if transform == "log1p":
        deltas = np.log1p(deltas)
    elif transform != "none":
        raise ValueError(f"Unsupported time_transform: {transform}")

    max_time_span = float(getattr(args, "max_time_span", 0.0) or 0.0)
    if max_time_span > 0:
        deltas = np.clip(deltas, 0.0, max_time_span)
    return deltas.tolist()


def _pad_left(values, max_len, pad_value):
    values = list(values)[-max_len:]
    return [pad_value] * (max_len - len(values)) + values


def _next_item_sample(seq, timestamps, max_len, args):
    tokens = seq[:-1][-max_len:]
    labels = seq[1:][-max_len:]
    if len(timestamps) >= 2:
        deltas = np.diff(timestamps)[-max_len:]
        deltas = _transform_deltas(deltas, args)
    else:
        deltas = []

    return (
        torch.LongTensor(_pad_left(tokens, max_len, 0)),
        torch.LongTensor(_pad_left(labels, max_len, 0)),
        torch.FloatTensor(_pad_left(deltas, max_len, 1.0)),
    )


def _eval_sample(seq, answer, timestamps, max_len, args):
    seq = seq[-max_len:]
    if len(timestamps) >= 2:
        # For evaluation, timestamps include the target timestamp, so diff length
        # matches the input sequence length.
        deltas = np.diff(timestamps)[-len(seq):]
        deltas = _transform_deltas(deltas, args)
    else:
        deltas = []

    return (
        torch.LongTensor(_pad_left(seq, max_len, 0)),
        torch.LongTensor(answer),
        torch.FloatTensor(_pad_left(deltas, max_len, 1.0)),
    )


class SASDataloader_ts(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        dataset = dataset.load_dataset_ts()
        self.train = dataset["train"]
        self.val = dataset["val"]
        self.test = dataset["test"]
        self.train_ts = dataset["train_ts"]
        self.val_ts = dataset["val_ts"]
        self.test_ts = dataset["test_ts"]
        self.umap = dataset["umap"]
        self.smap = dataset["smap"]
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        args.num_users = self.user_count
        args.num_items = self.item_count
        self.max_len = args.max_len
        self.sliding_size = args.sliding_window_size

    @classmethod
    def code(cls):
        return "sas"

    def get_pytorch_dataloaders(self):
        return self._get_train_loader(), self._get_val_loader(), self._get_test_loader()

    def _get_train_loader(self):
        dataset = SASTrainDatasetTS(self.args, self.train, self.train_ts, self.max_len, self.sliding_size)
        return data_utils.DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.args.num_workers,
            worker_init_fn=worker_init_fn,
        )

    def _get_val_loader(self):
        dataset = SASValidDatasetTS(self.args, self.train, self.val, self.train_ts, self.val_ts, self.max_len)
        return data_utils.DataLoader(
            dataset,
            batch_size=self.args.val_batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.args.num_workers,
        )

    def _get_test_loader(self):
        dataset = SASTestDatasetTS(
            self.args,
            self.train,
            self.val,
            self.test,
            self.train_ts,
            self.val_ts,
            self.test_ts,
            self.max_len,
        )
        return data_utils.DataLoader(
            dataset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.args.num_workers,
        )


class SASTrainDatasetTS(data_utils.Dataset):
    def __init__(self, args, u2seq, u2ts, max_len, sliding_size):
        self.args = args
        self.max_len = max_len
        self.sliding_step = max(1, int(sliding_size * max_len))
        self.all_seqs = []
        self.all_tss = []

        for user in sorted(u2seq.keys()):
            seq = u2seq[user]
            ts_seq = u2ts[user]
            if len(seq) != len(ts_seq):
                raise ValueError(f"Sequence and timestamp length mismatch for user {user}")

            window_len = self.max_len + 1
            if len(seq) <= window_len:
                self.all_seqs.append(seq)
                self.all_tss.append(ts_seq)
            else:
                start_idx = range(len(seq) - window_len, -1, -self.sliding_step)
                self.all_seqs.extend(seq[i : i + window_len] for i in start_idx)
                self.all_tss.extend(ts_seq[i : i + window_len] for i in start_idx)

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        return _next_item_sample(self.all_seqs[index], self.all_tss[index], self.max_len, self.args)


class SASValidDatasetTS(data_utils.Dataset):
    def __init__(self, args, u2seq, u2answer, u2ts, u2answer_ts, max_len):
        self.args = args
        self.u2seq = u2seq
        self.u2answer = u2answer
        self.u2ts = u2ts
        self.u2answer_ts = u2answer_ts
        self.users = [u for u in sorted(u2seq.keys()) if len(u2answer[u]) > 0]
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        timestamps = self.u2ts[user] + self.u2answer_ts[user]
        return _eval_sample(self.u2seq[user], self.u2answer[user], timestamps, self.max_len, self.args)


class SASTestDatasetTS(data_utils.Dataset):
    def __init__(self, args, u2seq, u2val, u2answer, u2ts, u2val_ts, u2answer_ts, max_len):
        self.args = args
        self.u2seq = u2seq
        self.u2val = u2val
        self.u2answer = u2answer
        self.u2ts = u2ts
        self.u2val_ts = u2val_ts
        self.u2answer_ts = u2answer_ts
        self.users = [u for u in sorted(u2seq.keys()) if len(u2val[u]) > 0 and len(u2answer[u]) > 0]
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2val[user]
        timestamps = self.u2ts[user] + self.u2val_ts[user] + self.u2answer_ts[user]
        return _eval_sample(seq, self.u2answer[user], timestamps, self.max_len, self.args)

