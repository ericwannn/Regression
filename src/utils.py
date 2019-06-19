import sys
import os
import pathlib
from time import time
from functools import wraps
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt


def timeit(func):
    @wraps(func)
    def _timeit(*args, **kwargs):
        time0 = time()
        func(*args, **kwargs)
        time1 = time()
        print("{} is called!".format(func.__name__))
        print("Time elapsed: %.3f s." % (time1 - time0))
    return _timeit


def echo(s):
    sys.stdout.write("\r%s" % s)
    sys.stdout.flush()


def get_leaf_file_names(directory):
    if pathlib.Path(directory).is_file():
        return [directory]
    else:
        return reduce(list.__add__, [
            get_leaf_file_names(directory + '/' + file_name)
            for file_name in os.listdir(directory)
        ])


def plot(y, y_hat):
    length = y.shape[0]
    x = np.linspace(3, 3*length, length)
    x_plot = x[:, np.newaxis]
    plt.scatter(x_plot, y, color='red')
    plt.plot(x_plot, y_hat, color='blue')
    plt.show()
