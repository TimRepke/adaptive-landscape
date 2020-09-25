from externals.umap.umap import UMAP
from datasets.mnist import MNIST
from datasets.fashion_mnist import FashionMNIST

import numpy as np
from data_handlers.generators import temporal, SamplingReference, accumulate
from data_handlers.disk import store_result
from models.large_vis import LargeVisModel
from models.fitsne import FItSNEModel
from evaluation.displacement import calculate_displacement_score
from data_handlers.plot import plot_grid
import os
from glob import glob
import logging
from util.log import init_logging

init_logging(['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'][4])
logger = logging.getLogger('main')

TEMP_FOLDER = '../TempData'
OUTPUT_FOLDER = '../OutputData'

logger.info('> Loading MNIST datasets...')
DATASETS = [
    ('mnist', MNIST('../RawData/MNIST')),
    # ('fashion_mnist', FashionMNIST('../RawData/FashionMNIST'))
]

LABEL_DIST = [{3: 0.005}, {3: 0.01}, {3: 0.1}]
SAMPLING_REF = SamplingReference.LABEL_COUNT
TARGET_SIZE = 10000

umap_settings = {
    'n_neighbors': 15,
    'n_components': 2,
    'metric': "euclidean",
    'metric_kwds': None,
    'output_metric': "euclidean",
    'output_metric_kwds': None,
    'n_epochs': None,
    'learning_rate': 1.0,
    'init': "spectral",
    'min_dist': 0.1,
    'spread': 1.0,
    'low_memory': False,
    'set_op_mix_ratio': 1.0,
    'local_connectivity': 1.0,
    'repulsion_strength': 1.0,
    'negative_sample_rate': 5,
    'transform_queue_size': 4.0,
    'a': None,
    'b': None,
    'random_state': None,
    'angular_rp_forest': False,
    'target_n_neighbors': -1,
    'target_metric': "categorical",
    'target_metric_kwds': None,
    'target_weight': 0.5,
    'transform_seed': 42,
    'force_approximation_algorithm': False,
    'verbose': False,
    'unique': False
}
for dataset_name, dataset in DATASETS:
    logger.info(f'> Preparing fake dynamic data for {dataset_name}...')
    temporal_data, temporal_labels = temporal(dataset,
                                              label_dist=LABEL_DIST,
                                              target_size=TARGET_SIZE,
                                              sampling_reference=SAMPLING_REF)
    umap_fix = UMAP(**umap_settings)
    umap_semi_fix = UMAP(**umap_settings)
    umap_flex = None
    Ys_fix = []
    Ys_semi_fix = []
    Ys_flex = []
    labs = []
    for interval, (interval_data, interval_labels) in enumerate(
            accumulate(temporal_data, temporal_labels, generate=True)):
        logger.info(f'> Running UMAP for interval {interval} with {len(interval_data)} data points.')
        if interval == 0:
            logger.info(' - Fitting umap_fix')
            umap_fix.fit(interval_data)
        logger.info(' - Fitting umap_semi_fix')
        umap_semi_fix.fit(interval_data)

        logger.info(' - Appending umap_fix')
        Ys_fix.append(umap_fix.transform(interval_data))
        logger.info(' - Appending umap_semi_fix')
        Ys_semi_fix.append(umap_semi_fix.transform(interval_data))

        if interval == 0:
            Ys_flex.append(Ys_fix[0])
            umap_flex = umap_fix
        if interval > 0:
            logger.info(' - Init umap_flex')
            umap_settings['init'] = umap_flex.transform(interval_data)
            umap_flex = UMAP(**umap_settings)
            logger.info(' - Fitting umap_flex')
            umap_flex.fit(interval_data)
            logger.info(' - Appending umap_flex')
            Ys_flex.append(umap_flex.transform(interval_data))
            print(len(Ys_flex))

        logger.info(' - Appending labels')
        labs.append(np.array([int(l) for l in interval_labels]))

    logger.info(' - Plotting umap_fix')
    plot_grid(target_file=f'{OUTPUT_FOLDER}/{dataset_name}_umap_consecutive_fix.pdf',
              data=Ys_fix, labels=labs, desc=list(range(len(labs))), rows=1, cols=3, figsize=(5, 2), point_size=0.1)
    logger.info(' - Plotting umap_semi_sfix')
    plot_grid(target_file=f'{OUTPUT_FOLDER}/{dataset_name}_umap_consecutive_semi_fix.pdf',
              data=Ys_semi_fix, labels=labs, desc=list(range(len(labs))), rows=1, cols=3, figsize=(5, 2),
              point_size=0.1)
    logger.info(' - Plotting umap_flex')
    plot_grid(target_file=f'{OUTPUT_FOLDER}/{dataset_name}_umap_consecutive_flex.pdf',
              data=Ys_flex, labels=labs, desc=list(range(len(labs))), rows=1, cols=3, figsize=(5, 2), point_size=0.1)