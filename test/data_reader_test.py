from src.data_reader import DataReader
from src.data_reader import Dataset
from src.utils import get_leaf_file_names


def read_file_by_stock_code_and_date_test(data_reader):
    data = data_reader.get_stock_info(stock_code='000006', date='20180802')
    print(data.head())
    print(data.shape)
    print(data.columns)
    print(data.iloc[2399:2402])


def get_leaf_file_test(data_reader):
    print(data_reader.data_path)
    all_paths = data_reader.get_all_file_path()
    print(all_paths)
    print(len(all_paths))


def create_dataset_test(dataset):

    dataset.date_dataset_generator(output_directory='../output/',
                                   num_workers=8, testset_size=.4)


if __name__ == '__main__':
    dr = DataReader(
        data_path='../data/rawdata'
    )
    # read_file_by_stock_code_and_date_test(data_reader)

    ds = Dataset(
        data_path='../data/rawdata/')

    # create_dataset_test(ds)

    ds.parse_data_by_date()

