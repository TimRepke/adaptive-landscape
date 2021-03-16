from datasets.mnist import MNIST
from datasets.fashion_mnist import FashionMNIST
from datasets.newsgroups import Newsgroups

import numpy as np
from data_handlers.generators import temporal, SamplingReference, accumulate
from data_handlers.disk import ResultWriter, RawDataWriter
from evaluation.displacement import calculate_displacement_score
from evaluation.grid import test
from data_handlers.plot import plot_grid
import os
from glob import glob
import logging
from util.log import init_logging
import Code.externals.umap_git.umap as umap
import matplotlib.pyplot as plt
from data_handlers.plot import COLORS_11, COLORS_12, COLORS_16, COLORS_39

init_logging(['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'][4])
logger = logging.getLogger('main')

TEMP_FOLDER = '../TempData'
OUTPUT_FOLDER = '../OutputData'

logger.info('> Loading MNIST datasets...')
DATASETS = [
    # ('20news', Newsgroups('../RawData/20news', f'{TEMP_FOLDER}/20news/parsed')),
    ('mnist', MNIST('../RawData/MNIST')),
    # ('fashion_mnist', FashionMNIST('../RawData/FashionMNIST'))
]
# autofill = dataset_name != '20news'
autofill = False
# LABEL_DIST = [{3: 0.001}, {3: 0.01}, {3: 0.1}, {3: 0.2}]
LABEL_DIST = [
    {l: 1 if l in {4, 6} else 200 for l in range(10)},
    {4: 200},
    {7: 400},
    {6: 400}
]
# LABEL_DIST = [{3: 0.001}, {3: 0.005}, {3: 0.01}, {3: 0.05}, {3: 0.1}]
# LABEL_DIST = [
#    {l: 800 if l != 1 else 1 for l in range(20)},
#    {1: 100},
#    {1: 300},
#    {1: 600}
# ]
# LABEL_DIST = [
#    {l: 200 if l != 1 else 1 for l in range(20)},
#    {1: 10},
#    {1: 50},
#    {1: 140}
# ]
SAMPLING_REF = SamplingReference.LABEL_COUNT
TARGET_SIZE = 2000


def axis_bounds(embedding):
    left, right = embedding.T[0].min(), embedding.T[0].max()
    bottom, top = embedding.T[1].min(), embedding.T[1].max()
    adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1
    return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]


all_in_one = True
if __name__ == '__main__':
    for dataset_name, dataset in DATASETS:
        logger.info(f'> Preparing fake dynamic data for {dataset_name}...')
        dataset.load()
        temporal_data, temporal_labels = temporal(dataset,
                                                  label_dist=LABEL_DIST,
                                                  target_size=TARGET_SIZE,
                                                  sampling_reference=SAMPLING_REF,
                                                  auto_fill=autofill)
        intervals = list(accumulate(temporal_data, temporal_labels))
        data = [d[0] for d in intervals]
        labels = [d[1] for d in intervals]

        relation_dicts = [{i: i for i in range(len(l))}
                          for l in labels[:-1]]

        if all_in_one:
            logger.debug('running all at once')
            aligned_mapper = umap.AlignedUMAP(verbose=True,
                                              n_neighbors=8).fit(data, relations=relation_dicts)
            points = aligned_mapper.embeddings_
        else:
            logger.debug('running with updates')
            updating_mapper = umap.AlignedUMAP(verbose=True).fit(data[:2], relations=relation_dicts[:1])
            for i in range(2, len(data)):
                logger.debug(f'loop {i}')
                updating_mapper.update(data[i], relations={v: k for k, v in relation_dicts[i - 1].items()})
            points = updating_mapper.embeddings_
        logger.info('Done, plotting now.')
        fig, axs = plt.subplots(1, len(labels), figsize=(10 * len(labels), 10))
        ax_bound = axis_bounds(np.vstack(points))
        cmap = COLORS_39 if dataset_name == '20news' else COLORS_16
        for i, ax in enumerate(axs.flatten()):
            # current_target = ordered_target[150 * i:min(ordered_target.shape[0], 150 * i + 400)]
            current_target = labels[i]
            labels_ = np.unique(current_target)
            ax.scatter(*points[i].T, s=2, c=[cmap[li] for li in current_target])
            ax.axis(ax_bound)
            ax.set(xticks=[], yticks=[])
            for label in labels_:
                ax.text(np.mean(points[i][current_target == label, 0]),
                        np.mean(points[i][current_target == label, 1]), label,
                        ha='center', va='center')
        plt.tight_layout()
        plt.show()
