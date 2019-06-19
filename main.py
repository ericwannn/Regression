from __future__ import print_function
from __future__ import division

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from src.data_reader import Dataset
from src.model import Model
from src.utils import timeit
from src.utils import get_leaf_file_names


@timeit
def main(generate_data=False, save_result=True):
    # Global config
    _RAW_DATA_PATH = './data/rawdata'
    _DATA_PATH = './data/date_data/'
    _OUTPUT_DIR = './data/output/'

    # Training configuration
    _TEST_SET_SIZE = .4
    _SLIDING_WINDOW_SIZE = 5
    _CHUNK_SIZE = 3901 * 2
    _MODEL_RIDGE = Ridge
    _MODEL_KKR = KernelRidge
    _PARAMS_RIDGE = {
        'alpha': 1e-3,
        'random_state': 100
    }
    _PARAMS_KERNEL_RIDGE = {
        'alpha': 2.5,
        'random_state': 100
    }

    # Data configuration
    _TIME_FRAME_SIZE_Y = (1, 5, 15, 30)
    _TIME_FRAME_SIZE_X = (1, 5, 15)
    _X_1 = [
        'avg_delta_bid_size_1', 'avg_delta_bid_size_5', 'avg_delta_bid_size_15',
        'avg_delta_ask_size_1', 'avg_delta_ask_size_5', 'avg_delta_ask_size_15',
        'bid_size_proportion_1', 'bid_size_proportion_5', 'bid_size_proportion_15'
    ]
    _X_2 = [
        'buy_volume_1', 'buy_volume_5', 'buy_volume_15', 'sell_volume_1', 'sell_volume_5',
        'sell_volume_15', 'proportion_volume_1', 'proportion_volume_5', 'proportion_volume_15'
    ]
    _Y_1, _Y_2, _Y_3, _Y_4 = ['return_1_min', 'return_5_min', 'return_15_min', 'return_30_min']

    _COLUMNS_TO_NORMALIZE = [
        'avg_delta_bid_size_1', 'avg_delta_bid_size_5', 'avg_delta_bid_size_15',
        'avg_delta_ask_size_1', 'avg_delta_ask_size_5', 'avg_delta_ask_size_15',
        'buy_volume_1', 'buy_volume_5', 'buy_volume_15', 'sell_volume_1', 'sell_volume_5', 'sell_volume_15'
    ]

    # Parse raw data, calculate features and split by date.
    if generate_data:
        date_dataset = Dataset(data_path=_RAW_DATA_PATH,
                               time_frame_sizes_y=_TIME_FRAME_SIZE_Y,
                               time_frame_sizes_x=_TIME_FRAME_SIZE_X)
        date_dataset.parse_data_by_date(num_workers=8, output_directory=_DATA_PATH)

    # Prepare dataset
    date_files = get_leaf_file_names(_DATA_PATH)
    data_train = date_files[:-int(len(date_files) * _TEST_SET_SIZE)]
    data_test = date_files[-int(len(date_files) * _TEST_SET_SIZE) - _SLIDING_WINDOW_SIZE + 1:]

    # Train model with different parameters
    model = Model(
        X=_X_2, y=_Y_1, model=_MODEL_RIDGE, params=_PARAMS_RIDGE,
        data_files=data_train, columns_to_normalize=_COLUMNS_TO_NORMALIZE,
        window_size=_SLIDING_WINDOW_SIZE
    )
    corrcoef, y_preds, y_truth = model.run()

    # if save_result:
    #     with open(_OUTPUT_DIR+'y_preds.pkl', 'wb') as file:
    #         pickle.dump(y_preds, file)
    #     with open(_OUTPUT_DIR+'y_truth.pkl', 'wb') as file:
    #         pickle.dump(y_truth, file)


if __name__ == '__main__':
    main(generate_data=False, save_result=True)
