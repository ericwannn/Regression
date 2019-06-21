from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd


class Transaction(object):
    """ Parse one file, calculate respective features including:
    1. return over certain time_frame_size, i.e. 1 min, 5 min and etc. as `y`
    2. average of delta-of-bid/ask-size-related features as `x1`
    3. volume initiated by buy/sell as `x2`
    4. more in the future ;)
    """
    def __init__(
            self, data, bid_size_column='BidSize',
            bid_price_column='BidPX', ask_size_column='OfferSize',
            ask_price_column='OfferPX', last_price_column='LastPx',
            total_volume_column='TotalVolumeTrade', total_value_column='TotalValueTrade'):
        self.data = data
        self.price = self.data[last_price_column]
        self.bid_size_column_names = [x for x in self.data.columns if x.startswith(bid_size_column)]
        self.bid_price_column_names = [x for x in self.data.columns if x.startswith(bid_price_column)]
        self.ask_size_column_names = [x for x in self.data.columns if x.startswith(ask_size_column)]
        self.ask_price_column_names = [x for x in self.data.columns if x.startswith(ask_price_column)]
        self.total_volume_column = total_volume_column
        self.total_value_column = total_value_column

    def parse(self, time_frame_sizes_y=(1, 5, 15, 30),
              time_frame_sizes_x=(1, 5, 15), truncate_inf=True):
        rtn = self.get_return(time_frame_sizes=time_frame_sizes_y)
        avg_delta = self.get_average_delta_bid_ask_size(time_frame_sizes=time_frame_sizes_x)
        volume = self.get_volume_init_by_buy_and_sell(time_frame_sizes=time_frame_sizes_x)
        lag_return = self.get_lag_return(time_frame_sizes=time_frame_sizes_x)

        # In this case, the valid range is [300, 4201)
        valid_from = self.minute_to_n_rows(max(time_frame_sizes_x)) + 1  # 300 in this case
        valid_till = - self.minute_to_n_rows(max(time_frame_sizes_y))  # -600, which is 4201 in this case.

        result = pd.concat([rtn, avg_delta, volume, lag_return], axis=1)
        if truncate_inf:
            self._truncate_inf(result)
        return result.iloc[valid_from: valid_till]

    ############
    # Find `y` #
    ############

    def get_return(self, time_frame_sizes=(1, 5, 15, 30)):
        result = pd.concat([self._get_return_by_time_frame_size(size) for size in time_frame_sizes], axis=1)
        result.columns = ['return_{}_min'.format(size) for size in time_frame_sizes]
        return result

    def _get_return_by_time_frame_size(self, time_frame_size):
        return self.price.shift(- self.minute_to_n_rows(time_frame_size)) / self.price - 1

    ###################
    # Find lag return #
    ###################

    def get_lag_return(self, time_frame_sizes=(1, 5, 15)):
        result = pd.concat([self._get_lag_return_by_time_frame_size(size) for size in time_frame_sizes], axis=1)
        result.columns = ['lag_return_{}'.format(size) for size in time_frame_sizes]
        return result

    def _get_lag_return_by_time_frame_size(self, time_frame_size):
        return self.price / self.price.shift(self.minute_to_n_rows(time_frame_size)) - 1

    ###################################################################
    # Find average of delta of bid/ask size and respective proportion #
    ###################################################################

    def get_average_delta_bid_ask_size(self, time_frame_sizes=(1, 5, 15)):

        bid_column_names = ['avg_delta_bid_size_{}'.format(size) for size in time_frame_sizes]
        ask_column_names = ['avg_delta_ask_size_{}'.format(size) for size in time_frame_sizes]
        proportion_column_names = ['bid_size_proportion_{}'.format(size) for size in time_frame_sizes]

        bid_size_sum = self.data[self.bid_size_column_names].sum(axis=1)
        ask_size_sum = self.data[self.ask_size_column_names].sum(axis=1)

        # get avg delta of bid size
        delta_bid_size = bid_size_sum - bid_size_sum.shift(1)
        avg_delta_bid_size = self._find_average(delta_bid_size, time_frame_sizes)

        # get avg delta of ask size
        delta_ask_size = ask_size_sum - ask_size_sum.shift(1)
        avg_delta_ask_size = self._find_average(delta_ask_size, time_frame_sizes)

        # get proportion of bid size
        proportion = bid_size_sum / (bid_size_sum + ask_size_sum)
        avg_proportion = self._find_average(proportion, time_frame_sizes)

        avg_proportion.columns = proportion_column_names
        avg_delta_ask_size.columns = ask_column_names
        avg_delta_bid_size.columns = bid_column_names

        result = pd.concat([avg_delta_bid_size, avg_delta_ask_size, avg_proportion], axis=1)
        return result

    ###################################
    # Find volume init by buy or sell #
    ###################################

    def get_volume_init_by_buy_and_sell(self, time_frame_sizes=(1, 5, 15, 30)):

        buy_volume_column_name = ["buy_volume_{}".format(size) for size in time_frame_sizes]
        sell_volume_column_name = ["sell_volume_{}".format(size) for size in time_frame_sizes]
        proportion_volume_column_name = ["proportion_volume_{}".format(size) for size in time_frame_sizes]

        avg_px = self._find_avg_price()

        # Get resp. original values
        sell = self._get_volume_init('sell', avg_px)
        buy = self._get_volume_init('buy', avg_px)
        proportion = buy.div(buy.add(sell)).fillna(.5)

        # Get average
        sell = self._find_average(sell, time_frame_sizes)
        buy = self._find_average(buy, time_frame_sizes)
        proportion = self._find_average(proportion, time_frame_sizes)

        # Rename
        sell.columns = sell_volume_column_name
        buy.columns = buy_volume_column_name
        proportion.columns = proportion_volume_column_name

        result = pd.concat([buy, sell, proportion], axis=1)

        return result

    def _get_volume_init(self, target, avg_px):
        if target == 'sell':
            volume_size = self.data[self.bid_size_column_names].to_numpy()
            valid_volume = self.data[self.bid_price_column_names].sub(
                avg_px, axis='rows').applymap(lambda x: 1 if x >= 0 else 0).to_numpy()

        else:
            volume_size = self.data[self.ask_size_column_names].to_numpy()
            valid_volume = self.data[self.ask_price_column_names].sub(
                avg_px, axis='rows').applymap(lambda x: 1 if x <= 0 else 0).to_numpy()

        return pd.DataFrame((valid_volume * volume_size).sum(axis=1))

    def _find_avg_price(self):
        return (self.data[self.total_value_column].shift(-1) - self.data[self.total_value_column]) / (
            self.data[self.total_volume_column].shift(-1) - self.data[self.total_volume_column])

    ####################
    # Helper functions #
    ####################

    def _find_average(self, data_frame, time_frame_sizes, axis=1):
        """ Consume a data frame, calculate the average according to the time frame size
        Note: current snapshot included.
        :param data_frame: say N*1, N records with M features
        :param time_frame_sizes: a list of length L with all the time frame size to cal avg
        :param axis: the direction to calculate average. 1 for averaging over a past period
        :return: a n*L pandas DataFrame. Each columns is the result of one average
        """
        assert axis in [1, -1]
        avg_df = data_frame.copy()
        global_shift = 1
        result = list()
        for size in time_frame_sizes:
            n_rows = self.minute_to_n_rows(size)
            while global_shift < n_rows:
                avg_df += data_frame.shift(axis * global_shift)
                global_shift += 1
            result.append(avg_df / n_rows)

        return pd.concat(result, axis=1)

    @staticmethod
    def minute_to_n_rows(minute):
        return int(60 * minute / 3)

    @staticmethod
    def _truncate_inf(data_frame, how='01'):
        """
        Truncate +/- np.inf values.
        :param data_frame:
        :param how: '01' turn inf and -inf into 1 and 0 resp.
                Otherwise, replace inf with the second greatest. -inf vice versa.
        :return: No return. Replace is inplace
        """
        if how == '01':
            data_frame.replace(-np.inf, 0, inplace=True)
            data_frame.replace(np.inf, 1, inplace=True)
        else:
            columns_with_inf = set(list(
                data_frame.columns[np.any(data_frame == -np.inf, axis=0)]
            ) + list(data_frame.columns[np.any(data_frame == np.inf, axis=0)]))
            for col in columns_with_inf:
                tmp = data_frame[col].replace([-np.inf, np.inf], data_frame[col].median())
                col_min, col_max = min(tmp), max(tmp)
                data_frame[col].replace(-np.inf, col_min, inplace=True)
                data_frame[col].replace(np.inf, col_max, inplace=True)
