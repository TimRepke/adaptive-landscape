import numpy as np
import functools
import operator
import gzip
import struct
import array
from matplotlib import pyplot as plt
from datasets import DataSet


class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass


# copied from https://github.com/datapythonista/mnist/blob/master/mnist/__init__.py
def parse_idx(fd):
    """Parse an IDX file, and return it as a numpy array.
    Parameters
    ----------
    fd : file
        File descriptor of the IDX file to parse
    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file
    1. https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, '
                             'file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, '
                             'file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type '
                             '0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             f'Expected: {expected_items}. Found: {len(data)}')

    return np.array(data).reshape(dimension_sizes)


# TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  60000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label
# ........
# xxxx     unsigned byte   ??               label
# The labels values are 0 to 9.
#
# TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  60000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ........
# xxxx     unsigned byte   ??               pixel
# Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
#
# TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  10000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label
# ........
# xxxx     unsigned byte   ??               label
# The labels values are 0 to 9.
#
# TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  10000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ........
# xxxx     unsigned byte   ??               pixel
# Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

class MNIST(DataSet):
    def __init__(self, raw_data_dir, flatten=True):
        super().__init__(raw_data_dir)
        with gzip.open(f'{self.raw_data_dir}/train-images-idx3-ubyte.gz', 'rb') as f:
            train_images = parse_idx(f)
        with gzip.open(f'{self.raw_data_dir}/train-labels-idx1-ubyte.gz', 'rb') as f:
            train_labels = parse_idx(f)
        with gzip.open(f'{self.raw_data_dir}/t10k-images-idx3-ubyte.gz', 'rb') as f:
            test_images = parse_idx(f)
        with gzip.open(f'{self.raw_data_dir}/t10k-labels-idx1-ubyte.gz', 'rb') as f:
            test_labels = parse_idx(f)

        self.images = np.vstack((train_images, test_images))
        if flatten:
            self.images = self.images.reshape((self.images.shape[0], -1))
        self.labels = np.hstack((train_labels, test_labels)).astype(int)

    def __len__(self):
        return len(self.get_data())

    @property
    def data_labels(self):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def generate_data(self):
        for i in self.get_data():
            yield i

    def get_data(self):
        return self.images

    def get_labels(self):
        return self.labels

    def generate_labels(self):
        for l in self.get_labels():
            yield l

    def show(self, sample):
        plt.imshow(sample, cmap='gray')
        plt.show()
