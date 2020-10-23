from abc import ABC, abstractmethod
from typing import Callable, Iterable
import logging
import hnswlib
import numpy as np

logger = logging.getLogger('model')


class Model(ABC):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def fit_data(cls, data=None, params=None):
        pass

    @classmethod
    def strategy_static(cls, data=None, params=None, strategy_params=None):
        """"This strategy blindly calculates a fresh dim-red-embedding
            for each interval.
        """
        for interval, (interval_data, interval_labels) in enumerate(data):
            logger.info(f'> Running {cls.__name__} (static) for interval {interval} '
                        f'with {len(interval_data)} data points.')
            yield interval_labels, cls.fit_data(interval_data, params)

    @classmethod
    def fit_intervals(cls, strategy: Callable, data: Iterable, params, **kwargs):
        strategy(data, params, **kwargs)

    @classmethod
    def get_neighbourhoods(cls, prev_hd, curr_hd, k=5):
        logger.debug(f'Building neighbourhood index for k={k} with {prev_hd.shape} items '
                     f'to look up {curr_hd.shape} items...')
        # Declaring index
        p = hnswlib.Index(space='l2', dim=prev_hd.shape[1])  # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        p.init_index(max_elements=prev_hd.shape[0], ef_construction=200, M=16)
        p.add_items(prev_hd)
        # Controlling the recall by setting ef:
        p.set_ef(2 * k)  # ef should always be > k

        return p.knn_query(curr_hd, k=k)

    @classmethod
    def get_init_mean_weighted(cls, prev_2d, prev_hd, curr_hd, k):
        neighbourhoods, distances = cls.get_neighbourhoods(prev_hd, curr_hd, k)
        logger.debug('Calculating mean weighted points for next init...')
        return np.array([np.average(prev_2d[neighbours], axis=0,
                                    weights=(np.max(distances[i]) - distances[i]) + 0.000001)
                         for i, neighbours in enumerate(neighbourhoods)])

    @classmethod
    def get_init_mean(cls, prev_2d, prev_hd, curr_hd, k):
        neighbourhoods, distances = cls.get_neighbourhoods(prev_hd, curr_hd, k)
        logger.debug('Calculating mean points for next init...')
        return np.array([np.average(prev_2d[neighbours], axis=0)
                         for i, neighbours in enumerate(neighbourhoods)])

    @classmethod
    def get_init_median(cls, prev_2d, prev_hd, curr_hd, k):
        neighbourhoods, _ = cls.get_neighbourhoods(prev_hd, curr_hd, k)
        logger.debug('Calculating median points for next init...')
        return np.array([np.mean(prev_2d[neighbours], axis=0)
                         for neighbours in neighbourhoods])

    def export(self):
        pass
