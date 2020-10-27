import numpy as np
import logging
from evaluation import fit_gmms

logger = logging.getLogger('displacement')


def norm_landscape(data):
    return (data + np.abs(data.min(axis=0))) / (np.abs(data.min(axis=0)) + data.max(axis=0))


def calculate_displacement_score(interval_data, interval_labels):
    prev_data = None
    prev_labels = None
    ret = []
    for interval, (data, labels) in enumerate(zip(interval_data, interval_labels)):
        if prev_data is None:
            prev_data = norm_landscape(data)
            prev_labels = labels
            continue
        logger.info(f'Calculating displacement of points from interval {interval - 1} to {interval}')
        data = norm_landscape(data)
        displacement = np.linalg.norm(prev_data - data[:len(prev_data)], axis=1)
        logger.info(f'Number of compared points: {displacement.shape[0]}')

        ret.append({
            str(l): np.average(displacement[prev_labels == l])
            for l in np.unique(prev_labels)
        })
        prev_data = data
        prev_labels = labels
    return ret


def gaussian_displacement(interval_data, interval_labels):
    prev_gmms = None
    ret = []
    for interval, (data, labels) in enumerate(zip(interval_data, interval_labels)):
        if prev_gmms is None:
            prev_gmms = fit_gmms(data, labels)
            continue
        logger.info(f'Calculating gaussian overlaps of points from interval {interval - 1} to {interval}')
        gmms = fit_gmms(data, labels)

        ret.append({
            l: norm_prev[0].overlap(norm_curr[0]) * norm_prev[1].overlap(norm_curr[1])
            for l, (norm_prev, norm_curr) in enumerate(zip(prev_gmms[1:], gmms[1:]))
        })
        prev_gmms = gmms
    return ret


# TODO copy from thesne (loss function)
def movement_penalty(Ys, N):
    penalties = []
    for t in range(len(Ys) - 1):
        penalties.append(T.sum((Ys[t] - Ys[t + 1]) ** 2))

    return T.sum(penalties) / (2 * N)
