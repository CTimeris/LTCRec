import pickle
import shutil
import tempfile
import os
from pathlib import Path
import gzip
from abc import *
from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def preprocess_ts(self):
        pass

    @abstractmethod
    def load_ratings_df(self):
        pass

    @abstractmethod
    def maybe_download_raw_dataset(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def load_dataset_ts(self):
        self.preprocess_ts()
        dataset_path = self._get_preprocessed_dataset_path_ts()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset
    
    def remove_immediate_repeats(self, df):     # 去除重复数据
        df_next = df.shift()        # 向下平移一行
        is_not_repeat = (df['uid'] != df_next['uid']) | (df['sid'] != df_next['sid'])
        df = df[is_not_repeat]
        return df

    def filter_triplets(self, df):      # 去除交互少于min_uc的用户，和被交互少于min_sc的物品
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]
        return df
    
    def densify_index(self, df):        # 将离散的uid和sid重新映射为从1开始的索引。
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']), start=1)}
        smap = {s: i for i, s in enumerate(set(df['sid']), start=1)}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df, user_count):         # 转成序列数据，分成训练、验证、测试
        if self.args.split == 'leave_one_out':      # leave_one_out：-2验证 -1测试
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(
                lambda d: list(d.sort_values(by=['timestamp', 'sid'])['sid']))  # 每个用户记录，按时间戳和物品排序，转成列表
            train, val, test = {}, {}, {}
            for i in range(user_count):
                user = i + 1
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        else:
            raise NotImplementedError

    def split_df_ts(self, df, user_count):       # 保留时间戳
        if self.args.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            # 同时提取物品序列和时间戳序列（按时间戳排序）
            user2items_ts = user_group.progress_apply(
                lambda d: (
                    list(d.sort_values(by=['timestamp', 'sid'])['sid']),  # 物品序列
                    list(d.sort_values(by=['timestamp', 'sid'])['timestamp'])  # 时间戳序列
                )
            )
            train, val, test = {}, {}, {}
            train_ts, val_ts, test_ts = {}, {}, {}  # 时间戳序列
            for i in range(user_count):
                user = i + 1
                items, timestamps = user2items_ts[user]
                # 物品序列划分
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
                # 对应时间戳划分
                train_ts[user] = timestamps[:-2]
                val_ts[user] = timestamps[-2:-1] if len(timestamps) >= 2 else []
                test_ts[user] = timestamps[-1:] if len(timestamps) >= 1 else []
            return train, val, test, train_ts, val_ts, test_ts  # 新增返回时间戳
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)        # data

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())       # data/数据集名字

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()  # data/preprocessed
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')

    def _get_preprocessed_dataset_path_ts(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset_ts.pkl')

