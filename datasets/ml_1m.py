from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):      # 数据标识
        return 'ml-1m'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'movies.dat',
                'ratings.dat',
                'users.dat']

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        
        print("Raw file doesn't exist. Downloading...")
        tmproot = Path(tempfile.mkdtemp())
        tmpzip = tmproot.joinpath('file.zip')
        tmpfolder = tmproot.joinpath('folder')
        download(self.url(), tmpzip)
        unzip(tmpzip, tmpfolder)
        if self.zip_file_content_is_folder():
            tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
        shutil.move(tmpfolder, folder_path)
        shutil.rmtree(tmproot)
        print()

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()        # data/preprocessed/.../dataset_ts.pkl
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()         # 从data/ml-1m/ratings.dat读取df: u(uid), i(sid), 得分(rating), 时间戳
        df = self.remove_immediate_repeats(df)      # 去掉连续的重复交互
        df = self.filter_triplets(df)               # 过滤交互过少的用户和物品
        df, umap, smap = self.densify_index(df)     # 将原始uid、sid重新映射到1开始的索引。
        train, val, test = self.split_df(df, len(umap))     # 转成序列并分割训练、验证、测试

        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:      # 写成preprocessed中的pkl文件
            pickle.dump(dataset, f)

    def preprocess_ts(self):
        dataset_path = self._get_preprocessed_dataset_path_ts()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        df = self.remove_immediate_repeats(df)
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        # 接收包含时间戳的划分结果
        train, val, test, train_ts, val_ts, test_ts = self.split_df_ts(df, len(umap))

        dataset = {
            'train': train, 'val': val, 'test': test,
            'train_ts': train_ts, 'val_ts': val_ts, 'test_ts': test_ts,     # 时间戳
            'umap': umap, 'smap': smap
        }
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()       # data/ml-1m
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
