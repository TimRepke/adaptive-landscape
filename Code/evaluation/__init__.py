import numpy as np
import logging
from sklearn.mixture import GaussianMixture
from statistics import NormalDist

logger = logging.getLogger('eval')


def fit_gmms(data, labels):
    logger.info('Fitting Gaussian Mixture models...')
    norm_dists = []

    gmm = GaussianMixture(n_components=1, covariance_type='spherical')
    gmm.fit(data)
    norm_dists.append((NormalDist(mu=gmm.means_[0][0], sigma=gmm.covariances_[0]),
                       NormalDist(mu=gmm.means_[0][1], sigma=gmm.covariances_[0])))

    for l in sorted(np.unique(labels)):
        gmm = GaussianMixture(n_components=1, covariance_type='spherical')
        samples = data[labels == l]
        if len(samples) < 2:
            samples = np.array([samples[0], samples[0]])
        gmm.fit(samples)
        norm_dists.append((NormalDist(mu=gmm.means_[0][0], sigma=gmm.covariances_[0]),
                           NormalDist(mu=gmm.means_[0][1], sigma=gmm.covariances_[0])))

    return norm_dists
