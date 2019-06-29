from __future__ import print_function
from __future__ import division

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from src.data_reader import Dataset
from src.model import Model
from src.utils import timeit
from src.utils import get_leaf_file_names


@timeit
def main(generate_data=False, mode='Train'):
    assert mode in ['Train', 'Test']

    # Global config
    _RAW_DATA_PATH = './data/rawdata'
    _DATA_PATH = './data/date_data'
    _OUTPUT_DIR = './data/output/'
    _OUTPUT_MODEL = './output/models/model_x1_y2'

    # Training configuration
    _TEST_SET_SIZE = 1
    _SLIDING_WINDOW_SIZE = 2
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
    _X_3 = [
        'lag_return_1', 'lag_return_5', 'lag_return_15'
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

    date_files = get_leaf_file_names(_DATA_PATH)
    if mode == 'Train':
        # Prepare dataset
        data_train = date_files[: 3]

        # Train model with different parameters
        model = Model(
            X=_X_1+_X_2+_X_3, y=_Y_1, model=_MODEL_RIDGE, params=_PARAMS_RIDGE,
            output_model_name=_OUTPUT_MODEL,
            data_files=data_train, columns_to_normalize=_COLUMNS_TO_NORMALIZE,
            window_size=_SLIDING_WINDOW_SIZE, days_as_window=True
        )
        stats, corrcoef = model.run()
    else:
        data_test = date_files[- _TEST_SET_SIZE:]


if __name__ == '__main__':
    main(generate_data=False, mode='Train')
