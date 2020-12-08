import os
import re
from typing import Union, Optional
import numpy as np
from sklearn.datasets import make_blobs
import logging

from datasets import DataSet

logger = logging.getLogger('artificial')


class Blobs(DataSet):

    def __init__(self,
                 n_samples: Union[int, list, np.ndarray],
                 n_dimensions: int,
                 n_blobs: Optional[int] = None,
                 blob_std: Union[float, list[float]] = 1.0,
                 bounds: tuple[float, float] = (-10.0, 10.0),
                 rand_state: Optional[Union[int, np.RandomState]] = None):
        super().__init__(None)

        self.n_samples = n_samples
        self.n_dimensions = n_dimensions
        self.n_blobs = n_blobs
        self.blob_std = blob_std
        self.bounds = bounds
        self.rand_state = rand_state

        self.vecs, self.labels = None, None

    def load(self):
        self.vecs, self.labels = make_blobs(n_samples=self.n_samples, n_features=self.n_dimensions,
                                            centers=self.n_blobs, cluster_str=self.blob_std, center_box=self.bounds,
                                            shuffle=False, random_state=self.rand_state, return_centers=False)

    def __len__(self) -> int:
        return self.n_samples

    @property
    def data_labels(self) -> np.ndarray:
        # np.unique: sorted unique elements of an array.
        return np.unique(self.labels)

    def generate_data(self):
        for i in self.get_data():
            yield i

    def get_data(self):
        return self.vecs

    def generate_labels(self):
        for i in self.get_labels():
            yield i

    def get_labels(self):
        return self.labels
