from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd


class Model(object):
    """
    Given paths of files and necessary configuration.
    1. Create dataset with a sliding windows of size N, which contains several days (or numbers of records) of features.
    2. Use the first N - 1 days of data as training set. The N-th (last) day of data as validation set.
    3. Slide the window forward by one day. Go to step 1 and repeat till the window reaches the end of all training data
    """
    def __init__(self, X, y, model, params,
                 data_files, columns_to_normalize,
                 days_as_window=False,
                 window_size=3, chunk_size=3901):
        """
        :param X: features
        :param y: labels
        :param model: regression model
        :param params: model config
        :param data_files:  paths to the files
        :param columns_to_normalize: select certain columns to normalize
        :param days_as_window: when set to True, window slides on date dataset (each time slide 1 day at least)
        :param window_size: specify number of date data or records in one window
        :param chunk_size: when days_as_window is False, window slides several lines of record instead.
        """
        self.X = X
        self.y = y
        self._model = model
        self.params = params
        self.data_files = data_files
        self.columns_to_normalize = list(set(columns_to_normalize).intersection(self.X))
        self.columns_to_normalize.append(self.y)
        self.days_as_window = days_as_window
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.y_preds = []
        self.y_truth = []
        self.corrcoef = []

    def run(self):
        if self.days_as_window:
            g_dataset = self.date_dataset_generator()
            for (idx, files, train, val) in g_dataset:
                print('Training phrase {}...'.format(idx + 1))
                print('Training on {}'.format([x.split('/')[-1] for x in files[:-1]]))
                model = self.train(self._model, self.params, train)
                print('Validating on {}'.format(files[-1].split('/')[-1]))
                self.update_val_result(model, train, val)
            return self.corrcoef, self.y_preds, self.y_truth
        else:
            g_dataset = self.column_dataset_generator()
            for idx, train, val in g_dataset:
                print('Training phrase {}...'.format(idx + 1))
                model = self.train(self._model, self.params, train)
                self.update_val_result(model, train, val)
            print(sum(self.corrcoef) / len(self.corrcoef))
            return self.corrcoef, self.y_preds, self.y_truth

    def train(self, _model, params, data):
        _model = _model(**params)
        X_train = data[self.X]
        y_train = data[self.y]
        _model.fit(X_train, y_train)
        return _model

    def update_val_result(self, model, data_train, data_val):
        preds_on_train = model.predict(data_train[self.X])
        score_on_training_set = np.corrcoef(data_train[self.y], preds_on_train)
        print('Score on training set:', score_on_training_set)
        y_preds = model.predict(data_val[self.X])
        y_truth = data_val[self.y]
        self.y_preds.append(y_preds)
        self.y_truth.append(y_truth)
        corr = np.corrcoef(y_preds, y_truth)
        self.corrcoef.append(corr)
        print('Score on validation set:', corr)

    def column_dataset_generator(self):
        window_length = self.window_size * self.chunk_size
        for file in self.data_files:
            print('Training on {}...\n\n\n'.format(file.split('/')[-1]))
            df = self._file_reader(file)[0]
            for train_idx in range(0, df.shape[0] - window_length, self.chunk_size):
                data_train = df.iloc[train_idx: train_idx + window_length - self.chunk_size].copy()
                data_val = df.iloc[train_idx + window_length - self.chunk_size: train_idx + window_length].copy()
                train, val = self._pre_process(data_train, data_val)
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
            train_data, val_data = self._pre_process(train_data.copy(), val_data.copy())

            yield train_idx, files, train_data, val_data

    def _pre_process(self, train, val, bound=5):
        train = train - train.mean()
        val = val - val.mean()
        train_std = train[self.columns_to_normalize].std()
        val_std = val[self.columns_to_normalize].std()
        for column in self.columns_to_normalize:
            train.loc[:, column] = train[column] / train_std[column]
            train.loc[:, column] = train[column].apply(
                lambda x: bound if x > bound else -bound if x < -bound else x)

            val.loc[:, column] = val[column] / val_std[column]
            val.loc[:, column] = val[column].apply(
                lambda x: bound if x > bound else -bound if x < -bound else x)
        return train, val

    @staticmethod
    def _file_reader(file_names):
        if isinstance(file_names, list):
            return [pd.read_pickle(file) for file in file_names]
        else:
            return [pd.read_pickle(file_names)]
