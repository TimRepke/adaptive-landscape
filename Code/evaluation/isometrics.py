# Some Code for the metrics is inspired by
# externals/Markov_Lipschitz_Deep_Learning/utils.py
# externals/topological_autoencoders/src/evaluation/measures_optimized.py
# externals/topological_autoencoders/src/evaluation/measures.py
# externals/topological_autoencoders/src/evaluation/eval.py

from typing import Literal
from hnswlib import Index
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, squareform
import logging

logger = logging.getLogger('isometrics')


class Measurer:
    def __init__(self, hd_data: np.ndarray, ld_data: np.ndarray, labels: np.ndarray, k: int = 5,
                 knn_metric_hd: Literal['cosine', 'l2', 'ip'] = 'cosine',
                 knn_metric_ld: Literal['cosine', 'l2', 'ip'] = 'l2',
                 random_seed=100):
        self.random_seed = random_seed
        self.hd_data = hd_data
        self.ld_data = ld_data
        self.labels = labels
        self.k = k
        self.knn_metric_hd = knn_metric_hd
        self.knn_metric_ld = knn_metric_ld
        self.knn_index_hd = None
        self.knn_index_ld = None

        self.pairwise_dist_hd = None
        self.pairwise_dist_ld = None

        self.ranks_hd = None
        self.neighbourhoods_hd = None
        self.ranks_ld = None
        self.neighbourhoods_ld = None

    def _require_pairwise_dist(self):
        if self.pairwise_dist_hd is None:
            logger.info(f'Calculating pairwise distances for {len(self.hd_data)} high dimensional points ...')
            self.pairwise_dist_hd = squareform(pdist(self.hd_data))
        if self.pairwise_dist_ld is None:
            logger.info(f'Calculating pairwise distances for {len(self.ld_data)} 2-dimensional points...')
            self.pairwise_dist_ld = squareform(pdist(self.ld_data))

    def _require_neighbourhoods_ranks(self):
        """
        Inputs: 
        - distances,        distance matrix [n times n], 
        - k,                number of nearest neighbours to consider
        
        neighbourhood    contains the sample indices (from 0 to n-1) of kth nearest neighbor of current sample [n times k]
        ranks            contains the rank of each sample to each sample [n times n], whereas entry (i,j) gives the rank that sample j has to i (the how many 'closest' neighbour j is to i) 
        """
        self._require_pairwise_dist()
        if self.neighbourhoods_hd is None or self.ranks_hd is None:
            logger.info('Calculating neighbourhoods and ranks for high dimensional space...')
            indices = np.argsort(self.pairwise_dist_hd, axis=-1, kind='stable')
            # Extract neighbourhoods
            self.neighbourhoods_hd = indices[:, 1:self.k + 1]
            # Convert this into ranks (finally)
            self.ranks_hd = indices.argsort(axis=-1, kind='stable')
        if self.neighbourhoods_ld is None or self.ranks_ld is None:
            logger.info('Calculating neighbourhoods and ranks for 2-dimensional space...')
            indices = np.argsort(self.pairwise_dist_ld, axis=-1, kind='stable')
            # Extract neighbourhoods
            self.neighbourhoods_ld = indices[:, 1:self.k + 1]
            # Convert this into ranks (finally)
            self.ranks_ld = indices.argsort(axis=-1, kind='stable')

    def _require_knn_index(self):
        if self.knn_index_hd is None:
            logger.info(f'Building kNN tree for {len(self.hd_data)} high dimensional points...')
            self.knn_index_hd = self._construct_knn_index(self.knn_metric_hd, self.hd_data)
        if self.knn_index_ld is None:
            logger.info(f'Building kNN tree for {len(self.ld_data)} 2-dimensional points...')
            self.knn_index_ld = self._construct_knn_index(self.knn_metric_ld, self.ld_data)

    def _construct_knn_index(self, knn_metric, data):
        index = Index(space=knn_metric, dim=data.shape[1])

        # Initialize HNSW Index
        index.init_index(max_elements=len(data), ef_construction=200, M=16, random_seed=self.random_seed)

        # Build index tree from data

        index.add_items(data, num_threads=-1)

        # Set ef parameter for (ideal) precision/recall
        index.set_ef(2 * self.k)

        return index

    def _knn_query(self, data, space: Literal['ld', 'hd'], include_self=False):
        self._require_knn_index()

        k = self.k if include_self else (self.k + 1)
        index = self.knn_index_hd if space == 'hd' else self.knn_index_ld

        indices, distances = index.knn_query(data, k=k, num_threads=-1)

        labels = self.labels[indices]

        if include_self:
            return indices, distances, labels

        return indices[:, 1:], distances[:, 1:], labels[:, 1:]

    def _get_space_data(self, space: Literal['ld', 'hd']):
        if space == 'ld':
            return self.ld_data
        return self.hd_data

    def _majority_vote(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        max_count = max(counts)  # we choose randomized vote for ties
        winners = np.where(counts == max_count)
        vote_ind = np.random.choice(winners[0], 1)
        vote = unique[vote_ind]
        return vote

    def _get_majority_neighbourhood_labels(self, labels):
        return np.array([
            [self._majority_vote(neighbourhood[:k + 1])[0] for k in range(self.k)]
            for neighbourhood in labels
        ])

    def normalised_mutual_information(self, space: Literal['ld', 'hd']):
        """NMI score
        For each point, n=[1...k] nearest neighbours.
        For each n-neighbourhood determine the most frequent label.
        For each n-neighbourhood, calculate NMI-score across all points.
        :param space:
        :return:
        """
        data = self._get_space_data(space)
        indices, distances, labels = self._knn_query(data, space)
        neighbourhood_labels = self._get_majority_neighbourhood_labels(labels)
        return np.array([
            normalized_mutual_info_score(self.labels, neighbourhood_labels[:, k], average_method='arithmetic')
            for k in range(self.k)
        ])

    def accuracy(self, space: Literal['ld', 'hd']):
        """Accuracy
        For each point, n=[1...k] nearest neighbours.
        For each n-neighbourhood determine the most frequent label.
        For each n-neighbourhood, calculate accuracy-score across all points.
        :param space:
        :return:
        """
        data = self._get_space_data(space)
        indices, distances, labels = self._knn_query(data, space)
        neighbourhood_labels = self._get_majority_neighbourhood_labels(labels)
        return np.array([
            accuracy_score(self.labels, neighbourhood_labels[:, k], normalize=True)
            for k in range(self.k)
        ])

    def local_kullback_leibler(self, sigma=0.01):
        self._require_pairwise_dist()
        dists_hd = self.pairwise_dist_hd
        dists_hd /= np.max(dists_hd)
        dists_ld = self.pairwise_dist_ld
        dists_ld /= np.max(dists_ld)

        density_hd = np.sum(np.exp(-(dists_hd ** 2) / sigma), axis=-1)
        density_hd /= density_hd.sum(axis=-1)

        density_ld = np.sum(np.exp(-(dists_ld ** 2) / sigma), axis=-1)
        density_ld /= density_ld.sum(axis=-1)

        return (density_hd * (np.log(density_hd) - np.log(density_ld))).sum()

    def _trustworthiness(self, neighbourhoods_a, ranks_a, neighbourhoods_b, ranks_b,  n):
        """Calculates the trustworthiness measure between the data space `X`
             and the latent space `Z`, given a neighbourhood parameter `k` for
            defining the extent of neighbourhoods.
        """
        result = 0.0

        # Calculate number of neighbours that are in the $k$-neighbourhood of the latent space
        # but not in the $k$-neighbourhood of the data space
        for row in range(ranks_a.shape[0]):
            missing_neighbours = np.setdiff1d(neighbourhoods_b[row], neighbourhoods_a[row])

            for neighbour in missing_neighbours:
                result += (ranks_a[row, neighbour] - self.k)

        return 1 - 2 / (n * self.k * (2 * n - 3 * self.k - 1)) * result

    def continuity(self):
        """
        Calculates the continuity measure between the data space `X` and the
        latent space `Z`, given a neighbourhood parameter `k` for setting up
        the extent of neighbourhoods.

        This is just the 'flipped' variant of the 'trustworthiness' measure.
        :param k: 
        :return: 
        """
        self._require_neighbourhoods_ranks()
        n = self.pairwise_dist_hd.shape[0]

        return self._trustworthiness(self.neighbourhoods_ld, self.ranks_ld,
                                     self.neighbourhoods_hd, self.ranks_hd, n)

    def trustworthiness(self):
        self._require_neighbourhoods_ranks()
        n = self.pairwise_dist_hd.shape[0]

        return self._trustworthiness(self.neighbourhoods_hd, self.ranks_hd,
                                     self.neighbourhoods_ld, self.ranks_ld, n)
