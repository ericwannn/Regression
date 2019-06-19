from src.transaction import Transaction
from src.data_reader import DataReader


def return_test(trx):
    print('return function unit testing...')
    rtn = trx.get_return()
    print(rtn.head())
    print(rtn.shape)
    print(rtn.iloc[-25:])

    return rtn


def avg_delta_test(trx):
    print('get average delta of bid and ask unit testing...')
    rtn = trx.get_average_delta_bid_ask_size()
    print(rtn.head(30))
    print(rtn.shape)
    print(rtn.columns)

    return rtn


def volume_init_test(trx):
    rtn = trx.get_volume_init_by_buy_and_sell()
    print(rtn.head(30))
    print(rtn.shape)
    print(rtn.columns)
    print(rtn.describe())
    return rtn


if __name__ == '__main__':
    dr = DataReader(data_path='../data/rawdata')
    df = dr.get_stock_info(stock_code='000006', date='20180802')
    tr = Transaction(df)

    # return_test(tr)
    data = tr.parse()
    print(data.describe())
