from datasets.mnist import MNIST
from datasets.fashion_mnist import FashionMNIST

import numpy as np
from data_handlers.generators import temporal, SamplingReference, accumulate
from data_handlers.disk import store_result
from models.large_vis import LargeVisModel
from evaluation.displacement import calculate_displacement_score
import os
from glob import glob
import logging
from util.log import init_logging

# parser.add_argument('--log', type=str, default='DEBUG',
#                     choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
#                     help='Log Level')

init_logging(['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'][4])
logger = logging.getLogger('main')

TEMP_FOLDER = '../TempData'
OUTPUT_FOLDER = '../OutputData'

logger.info('> Loading MNIST dataset...')
m = MNIST('../RawData/MNIST')

# fm = FashionMNIST('../RawData/FashionMNIST')

_rerun = False

if _rerun:
    logger.info('> Preparing fake dynamic data...')
    temporal_data, temporal_labels = temporal(m,
                                              label_dist=[{3: 0.005}, {3: 0.01}, {3: 0.1}],
                                              target_size=10000,
                                              sampling_reference=SamplingReference.LABEL_COUNT)

    for interval, (interval_data, interval_labels) in enumerate(
            accumulate(temporal_data, temporal_labels, generate=True)):
        logger.info(f'> Running LargeVis for interval {interval} with {len(interval_data)} data points.')

        logger.info(' - Preparing temp data file...')
        temp_file = f'{TEMP_FOLDER}/lv_temp'
        LargeVisModel.write_temp_file(temp_file, interval_data)

        logger.info(' - Run LargeVis...')
        y = LargeVisModel.fit_interval(input_file=temp_file)

        logger.info(' - Storing results...')
        store_result(y, interval_labels, 'mnist', 'largevis', interval, OUTPUT_FOLDER)

        logger.info(' - Removing temp data file...')
        os.remove(temp_file)

_rerun = True
if _rerun:
    eval_files = glob(f'{OUTPUT_FOLDER}/mnist_largevis_*.tsv')
    calculate_displacement_score(eval_files)

# digits = [i[0].reshape((-1,)) for i in m.get_data()][:1000]
# labels = np.array([i[1] for i in m.get_data()][:1000])
# logger.info(digits[0])
#
# fdigits = [i[0].reshape((-1,)) for i in fm.get_data()][:1000]
# flabels = np.array([i[1] for i in fm.get_data()][:1000])
# logger.info()
# fm.show(fm.create_sprite_image())
# fm.show(fm.get_data()[0][0])
#
# # embedding = umap.UMAP(n_neighbors=5,
# #                      min_dist=0.3,
# #                      metric='correlation').fit_transform(m.get_data())
#
# mapper = umap.UMAP(n_neighbors=5,
#                    min_dist=0.1,
#                    metric='euclidean').fit(digits)
# umap.plot.points(mapper, labels=labels)
