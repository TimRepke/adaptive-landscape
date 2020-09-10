import os
import gzip
import numpy as np
from matplotlib import pyplot as plt

from data_readers import DataReader


class FashionMNIST(DataReader):
    def __init__(self, raw_data_dir):
        super().__init__(raw_data_dir)
        with gzip.open(f'{self.raw_data_dir}/train-labels-idx1-ubyte.gz', 'rb') as f:
            train_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        with gzip.open(f'{self.raw_data_dir}/train-images-idx3-ubyte.gz', 'rb') as f:
            train_images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(train_labels), 784)
        with gzip.open(f'{self.raw_data_dir}/t10k-labels-idx1-ubyte.gz', 'rb') as f:
            test_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        with gzip.open(f'{self.raw_data_dir}/t10k-images-idx3-ubyte.gz', 'rb') as f:
            test_images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(test_labels), 784)

        self.images = np.reshape(np.vstack((train_images, test_images)), (-1, 28, 28))
        self.labels = np.hstack((train_labels, test_labels))

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
        plt.imshow(sample, cmap='gray', interpolation='none', aspect='equal')
        plt.show()

    # https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/helper.py
    def create_sprite_image(self, invert=False):
        """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
        images = self.images
        if invert:
            images = 255 - images
        img_h = images.shape[1]
        img_w = images.shape[2]
        n_plots = int(np.ceil(np.sqrt(images.shape[0])))

        spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

        for i in range(n_plots):
            for j in range(n_plots):
                this_filter = i * n_plots + j
                if this_filter < images.shape[0]:
                    this_img = images[this_filter]
                    spriteimage[i * img_h:(i + 1) * img_h,
                    j * img_w:(j + 1) * img_w] = this_img

        return spriteimage
