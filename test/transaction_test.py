import numpy as np
from src.transaction import Transaction
from src.data_reader import DataReader


def return_test(trx):
    print('return function unit testing...')
    rtn = trx.get_return()
    print(rtn.shape)
    print(rtn.iloc[-602:-598])

    return rtn


def avg_delta_test(trx):
    print('get average delta of bid and ask unit testing...')
    rtn = trx.get_average_delta_bid_ask_size()
    print(rtn.iloc[298:302]['avg_delta_bid_size_15'])
    print(rtn.iloc[-602:-598]['avg_delta_ bid_size_15'])

    return rtn


def volume_init_test(trx):
    rtn = trx.get_volume_init_by_buy_and_sell()
    print(rtn.head(30))
    print(rtn.shape)
    print(rtn.columns)
    print(rtn.describe())
    return rtn


def get_lag_return_test(trx):
    rtn = trx.get_lag_return()
    print(rtn.iloc[298:302])
    print(rtn.shape)
    print(rtn.describe())


if __name__ == '__main__':
    dr = DataReader(data_path='../data/rawdata')
    df = dr.get_stock_info(stock_code='000006', date='20180820')
    tr = Transaction(df)
    data = tr.parse()
    print(np.isnan(data).any())
    print(data.shape)


