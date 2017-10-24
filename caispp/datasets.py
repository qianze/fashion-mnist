import os
import urllib.request
from pathlib import Path

# Data repositories
UCI_SEEDS_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'


# Download locations
UCI_SEEDS = 'uci_seeds.txt'

# First make sure the data folder exists.
home_dir = str(Path.home())
DATA_FOLDER_NAME = home_dir + '/.cais_data/'
if not os.path.exists(DATA_FOLDER_NAME):
    os.makedirs(DATA_FOLDER_NAME)


def __convert_to_num_array(line):
    line


def download_uci_seeds():
    if not os.path.exists(DATA_FOLDER_NAME + UCI_SEEDS):
        urllib.request.urlretrieve(UCI_SEEDS_URL, DATA_FOLDER_NAME + UCI_SEEDS)

    X = []
    Y = []
    with open(DATA_FOLDER_NAME + UCI_SEEDS) as f:
        for line in f:
            parts = line.split('\t')
            parts = filter(lambda x: x != '', parts)
            row = list(map(lambda x: float(x.replace('\n', '')), parts))
            X.append(row[0:7])
            Y.append(int(row[-1]))

    return X, Y

