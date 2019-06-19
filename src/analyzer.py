from __future__ import division
from __future__ import print_function

import multiprocessing as mp
from math import ceil
import pickle

import pandas as pd

from src.transaction import Transaction
from src.data_reader import DataReader


class Analyzer(object):
    def __init__(self, data_reader, time_frame_size_x=(1, 5, 15), time_frame_size_y=(1, 5, 15, 30)):
        self.data_reader = data_reader
        self.time_frame_size_x = time_frame_size_x
        self.time_frame_size_y = time_frame_size_y

    def batch_get_data_pairs(self, num_workers=4):
        all_file_path = self.data_reader.get_all_file_path(flatten=True)[:100]
        batch_size = int(ceil(len(all_file_path) / num_workers))
        pool = mp.Pool(processes=num_workers)
        small_batches = [
            all_file_path[idx: idx + batch_size]
            for idx in range(0, len(all_file_path), batch_size)
        ]
        res = pool.map(self.get_data_pairs, small_batches)
        pickle.dump(res, 'result.pkl')
        return res

    def get_data_pairs(self, file_path):
        transaction = Transaction(self.data_reader.get_stock_info(stock_file_path=file_path))
        turnover = transaction.get_return(self.time_frame_size_y)
        avg_delta_in_bid_and_ask = transaction.get_average_delta_bid_ask_size('all', self.time_frame_size_x)
        volume_init_by_buy_and_sell = transaction.get_volume_init_by_buy_and_sell('all', self.time_frame_size_x)
        result = [turnover] + avg_delta_in_bid_and_ask + volume_init_by_buy_and_sell
        return pd.concat(result, axis=1).iloc[:min(map(len, result))]


if __name__ == '__main__':
    dr = DataReader()
    analyzer = Analyzer(dr)
    res = analyzer.batch_get_data_pairs()
    print(res)
    print(type(res))

