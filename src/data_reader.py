from __future__ import division
from __future__ import print_function

import os
from math import ceil

import pandas as pd
from joblib import Parallel
from joblib import delayed

from src.transaction import Transaction
from src.utils import echo
from src.utils import get_leaf_file_names


class DataReader(object):

    def __init__(self,  data_path='../data/rawdata/'):
        self.data_path = data_path
        self.data_path = self.data_path

    def get_stock_info(self, stock_file_path=None, stock_code=None, date=None):
        # Two methods to fetch the record of one file:
        # 1. Specify the stock code and date
        # 2. Specify the path of the file
        # Then drop the 2400-th row to remove 11:30:00, which duplicates with 13:30:00
        file_path = stock_file_path if stock_file_path else self._expand_file_name(stock_code, date)
        echo('Processing {}...'.format(file_path))
        return self._file_handler(file_path).drop(2400, axis=0).reset_index()

    def get_all_file_path(self):
        return get_leaf_file_names(self.data_path)

    def _expand_file_name(self, stock_code, date):
        full_stock_code = stock_code + '.SZ' if (
                stock_code.startswith('00') or
                stock_code.startswith('200') or
                stock_code.startswith('300')
        ) else stock_code + '.SH'
        return '/'.join(
            [self.data_path, date, 'md', 'md_' + full_stock_code + '_' + date + '.csv.gz']
        )

    @staticmethod
    def _file_handler(file_path, compression='gzip'):
        try:
            file = pd.read_csv(file_path, compression=compression)
        except FileExistsError:
            print('{} not found! Skip and continue...')
            file = None
        return file


class Dataset(DataReader):

    def __init__(self, data_path,
                 time_frame_sizes_y=(1, 5, 15, 30),
                 time_frame_sizes_x=(1, 5, 15)):
        super(Dataset, self).__init__(data_path=data_path)
        self.time_frame_sizes_y = time_frame_sizes_y
        self.time_frame_sizes_x = time_frame_sizes_x

    def parse(self, stock_file_path=None, stock_code=None, date=None):
        data_frame = self.get_stock_info(stock_file_path=stock_file_path, stock_code=stock_code, date=date)
        return Transaction(data_frame).parse(
            time_frame_sizes_x=self.time_frame_sizes_x, time_frame_sizes_y=self.time_frame_sizes_y
        )

    def create_dataset(self, output_directory='../output/',
                       train_data_file_name='data_train.pkl',
                       test_data_file_name='test_data.pkl',
                       num_workers=4, testset_size=.4,
                       return_value=False):
        file_names = self.get_all_file_path()
        testset_len = int(len(file_names) * testset_size)
        train_files = file_names[:-testset_len]
        test_files = file_names[-testset_len:]
        data_train = self._create_dataset(train_files, num_workers, output_directory + train_data_file_name)
        print('Size of training set: %d' % len(data_train))
        if not return_value:
            del data_train
        data_test = self._create_dataset(test_files, num_workers, output_directory + test_data_file_name)
        print('Size of testing set: %d' % len(data_test))
        if not return_value:
            del data_test
        else:
            return data_train, data_test

    def _create_dataset(self, file_names, num_workers, output_file):
        batches = self._split_into_batches(file_names, num_workers)
        result = pd.concat(Parallel(n_jobs=num_workers)(
            delayed(self._parse_in_batch)(batch) for batch in batches))
        result.to_pickle(output_file)
        return result

    def _parse_in_batch(self, stock_file_paths):
        return pd.concat([self.parse(path) for path in stock_file_paths])

    def parse_data_by_date(self, num_workers=8, output_directory='../data/date_data/'):
        all_dates = [x.strip() for x in os.listdir(self.data_path) if x.startswith('2018')]
        for date in all_dates:
            file_names = get_leaf_file_names(self.data_path + '/' + date)
            _ = self._create_dataset(file_names, num_workers, output_directory + date + '.pkl')
        print('Date data created successfully in {}!'.format(output_directory))

    @staticmethod
    def _split_into_batches(file_names, num_workers):
        batch_size = int(ceil(len(file_names) / num_workers))
        return [
            file_names[idx: idx + batch_size]
            for idx in range(0, len(file_names), batch_size)
        ]


