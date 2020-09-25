from datasets.mnist import MNIST
from datasets.fashion_mnist import FashionMNIST

import numpy as np
from data_handlers.generators import temporal, SamplingReference, accumulate
from data_handlers.disk import store_result
from models.large_vis import LargeVisModel
from models.fitsne import FItSNEModel
from models.bhtsne import BHtSNEModel
from models.umap import UMAPModel
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

LABEL_DIST = [{3: 0.001}, {3: 0.01}, {3: 0.1}]
# LABEL_DIST = [{3: 0.001}, {3: 0.005}, {3: 0.01}, {3: 0.05}, {3: 0.1}]
SAMPLING_REF = SamplingReference.LABEL_COUNT
TARGET_SIZE = 20000


MODEL_CONFIGS = [
    (FItSNEModel, {'seed': 2020, 'df': 0.5}),
    (FItSNEModel, {'seed': 2020, 'df': 1.0}),
    (FItSNEModel, {'seed': 2020, 'df': 100.0}),
    (BHtSNEModel, {'randseed': 2020}),
    (UMAPModel, {}),
    (LargeVisModel, {}),
]

RUN_MODELS = not True
PLOT = not True
RUN_EVAL = True

if RUN_MODELS:
    for dataset_name, dataset in DATASETS:
        logger.info(f'> Preparing fake dynamic data for {dataset_name}...')
        temporal_data, temporal_labels = temporal(dataset,
                                                  label_dist=LABEL_DIST,
                                                  target_size=TARGET_SIZE,
                                                  sampling_reference=SAMPLING_REF)

        for interval, (interval_data, interval_labels) in enumerate(
                accumulate(temporal_data, temporal_labels, generate=True)):
            for mi, (model, config) in enumerate(MODEL_CONFIGS):
                logger.info(
                    f'> Running {model.__name__} for interval {interval} with {len(interval_data)} data points.')

                temp_file = None
                data = None
                if model is LargeVisModel:
                    logger.info(' - Preparing temp data file...')
                    temp_file = f'{TEMP_FOLDER}/lv_temp'
                    LargeVisModel.write_temp_file(temp_file, interval_data)
                else:
                    data = interval_data

                logger.info(f' - Run {model.__name__}...')
                logger.info(f'   Settings: {config}')
                y = model.fit_interval(data=data, input_file=temp_file, **config)

                logger.info(' - Storing results...')
                store_result(y, interval_labels, dataset_name, model.__name__ + str(mi), interval, OUTPUT_FOLDER)

                if model is LargeVisModel and temp_file is not None:
                    logger.info(' - Removing temp data file...')
                    os.remove(temp_file)

if PLOT:
    for dataset_name, _ in DATASETS:
        for mi, (model, _) in enumerate(MODEL_CONFIGS):
            eval_files = sorted(glob(f'{OUTPUT_FOLDER}/{dataset_name}_{model.__name__}{mi}*.tsv'))
            results = []
            labels = []
            for file in eval_files:
                logger.debug(f'Reading {file}...')
                with open(file, 'r') as f:
                    tmp = [
                        [float(li) for li in line.strip().split()]
                        for line in f
                    ]
                    labels.append(np.array([int(ti[-1]) for ti in tmp]))
                    results.append(np.array([[ti[0], ti[1]] for ti in tmp]))

            logger.info(f'Plotting results to {OUTPUT_FOLDER}/{dataset_name}_{model.__name__}{mi}.pdf')

            plot_grid(target_file=f'{OUTPUT_FOLDER}/{dataset_name}_{model.__name__}{mi}.pdf',
                      data=results, labels=labels, desc=eval_files, rows=1, cols=3, figsize=(5, 2), point_size=0.1)

if RUN_EVAL:
    for dataset_name, _ in DATASETS:
        for mi, (model, _) in enumerate(MODEL_CONFIGS):
            logger.info(f'Running evaluations for {dataset_name} and {model} ({mi})...')
            eval_files = glob(f'{OUTPUT_FOLDER}/{dataset_name}_{model.__name__}{mi}*.tsv')
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
