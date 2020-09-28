import numpy as np


def _read_file(file_name):
    with open(file_name, 'r') as f:
        return np.array([[float(dim) for dim in line.split()[:2]] + [int(line.split()[2])] for line in f])

