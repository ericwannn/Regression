from __future__ import print_function
from __future__ import division

import pickle
import pprint
import numpy as np
import pandas as pd


class Model(object):
    """
    Given paths of files and necessary configuration.
    1. Create dataset with a sliding windows of size N, which contains several days (or numbers of records) of features.
    2. Use the first N - 1 days of data as training set. The N-th (last) day of data as validation set.
    3. Slide the window forward by one day. Go to step 1 and repeat till the window reaches the end of all training data
    """
    def __init__(self, X, y, model, params, output_model_name,
                 data_files, columns_to_normalize,
                 centre_by='median', days_as_window=False,
                 window_size=3, chunk_size=3901):
        """
        :param X: features
        :param y: labels
        :param model: regression model
        :param params: model config
        :param data_files:  paths to the data files
        :param columns_to_normalize: select certain columns to normalize
        :param days_as_window: when set to True, window slides on date dataset (each time slide 1 day at least)
        :param window_size: specify number of date data or records in one window
        :param chunk_size: when days_as_window is False, window slides several lines of record instead.
        """
        self.X = X
        self.y = y
        self.model = model(**params)
        self.output_model_name = output_model_name
        self.data_files = data_files
        self.centre_by = centre_by
        self.columns_to_normalize = list(set(columns_to_normalize).intersection(self.X))
        # self.columns_to_normalize.append(self.y)
        self.days_as_window = days_as_window
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.corrcoef = []
        self.stats = []

    def run(self):
        if self.days_as_window:
            g_dataset = self.date_dataset_generator()
            for (idx, files, train, val) in g_dataset:
                print('------------------------------------------------------------')
                print('Training phrase {}...'.format(idx + 1))
                print('Training on {}'.format([x.split('/')[-1] for x in files[:-1]]))
                self.train(idx, train)
                print('Validating on {}'.format(files[-1].split('/')[-1]))
                self.update_val_result(train, val)
                print('------------------------------------------------------------\n')
            return self.stats, self.corrcoef
        else:
            g_dataset = self.column_dataset_generator()
            for idx, train, val in g_dataset:
                print('Training phrase {}...'.format(idx + 1))
                self.train(train)
                self.update_val_result(train, val)
            print(sum(self.corrcoef) / len(self.corrcoef))
            return self.corrcoef

    def train(self, idx, data):
        X_train = data[self.X]
        y_train = data[self.y]
        self.model.fit(X_train, y_train)
        with open('{}_{}.pickle'.format(self.output_model_name, idx), 'wb') as f:
            pickle.dump(self.model, f)

    def test(self, test_set, read_model=None):
        assert isinstance(test_set, list) and len(test_set) > 0
        model = read_model if read_model else self.model
        result = []
        for idx, data in enumerate(test_set):
            test_data = self.pre_process(self._file_reader(data)[0])
            y_hat = model.predict(test_data[self.X])
            y = test_data[self.y]
            stats = self.stat_analysis(y, y_hat, test_data, model.coef_)
            corr = np.corrcoef(y_hat, y)
            print('Corrcoef: {}'.format(corr))
            result.append({
                'y_hat': y_hat,
                'y': y,
                'corr': np.corrcoef(y_hat, y),
                'stats': stats
            })
        return result

    def update_val_result(self, data_train, data_val):
        preds_on_train = self.model.predict(data_train[self.X])
        score_on_training_set = np.corrcoef(data_train[self.y], preds_on_train)[0][1]
        print('Corrcoef on training set:', score_on_training_set)
        y_preds = self.model.predict(data_val[self.X])
        y_truth = data_val[self.y]
        corr = np.corrcoef(y_preds, y_truth)[0][1]
        self.corrcoef.append(corr)
        print('Corrcoef on validation set:', corr)
        stats = self.stat_analysis(y_truth, y_preds, data_val, self.model.coef_)
        self.stats.append(stats)

    def stat_analysis(self, y, y_hat, data, coefs):
        coefs = dict(zip(self.X, coefs))
        sse = np.square(y - y_hat).sum()
        s_2 = sse / (y.shape[0] - 2)
        ss_yy = np.square(y - y.mean()).sum()
        r_2 = 1 - sse / ss_yy
        t_value = [
            coefs[x] / np.sqrt(s_2 / np.square(data[x] - data[x].mean()).sum())
            for x in self.X
        ]
        print(t_value)
        stats = {
           'R_square': r_2,
           'stats': pd.DataFrame(zip(self.X, self.model.coef_, t_value), columns=['Feature', 'Coef', 'T-value'])
        }
        pprint.pprint(stats, indent=4)
        return stats

    def column_dataset_generator(self):
        window_length = self.window_size * self.chunk_size
        for file in self.data_files:
            print('Training on {}...\n\n\n'.format(file.split('/')[-1]))
            df = self._file_reader(file)[0]
            for train_idx in range(0, df.shape[0] - window_length, self.chunk_size):
                data_train = df.iloc[train_idx: train_idx + window_length - self.chunk_size].copy()
                data_val = df.iloc[train_idx + window_length - self.chunk_size: train_idx + window_length].copy()
                train, val = self.pre_process(data_train, data_val)
                yield train_idx, train, val

    def date_dataset_generator(self):
        files = self.data_files[:self.window_size]
        data_list = ['dummy'] + self._file_reader(files)
        files = ['dummy'] + files
        for train_idx in range(len(self.data_files) - self.window_size):
            # Update file path info
            files.pop(0)
            files.append(self.data_files[train_idx + self.window_size])

            # Move the window forward by one day
            data_list.pop(0)
            data_list += self._file_reader(self.data_files[train_idx + self.window_size])

            # Get train/validation data and pre-process some columns
            train_data = pd.concat(data_list[:-1])
            val_data = data_list[-1]
            train_data = self.pre_process(train_data)
            val_data = self.pre_process(val_data)
            yield train_idx, files, train_data, val_data

    def pre_process(self, data, bound=5):
        if self.centre_by == 'median':
            data -= data.median()
        else:
            data -= data.mean()
        std = data.std()
        for column in self.columns_to_normalize:
            data.loc[:, column] = data[column] / std[column]
            data.loc[:, column] = data[column].apply(
                lambda x: bound if x > bound else -bound if x < -bound else x
            )
        return data

    @staticmethod
    def _file_reader(file_names):
        print('Reading file {}'.format(file_names))
        if isinstance(file_names, list):
            return [pd.read_pickle(file).dropna(how='any') for file in file_names]
        else:
            return [pd.read_pickle(file_names).dropna(how='any')]
