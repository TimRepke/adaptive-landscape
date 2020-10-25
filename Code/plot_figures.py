from run_models import OUTPUT_FOLDER, MODEL_CONFIGS, DATASETS
from datasets.mnist import MNIST
from datasets.fashion_mnist import FashionMNIST
from datasets.newsgroups import Newsgroups

import numpy as np
from data_handlers.generators import temporal, SamplingReference, accumulate
from data_handlers.disk import ResultWriter, IntervalResultReader
from evaluation.displacement import calculate_displacement_score
from evaluation.grid import test
from data_handlers.plot import plot_grid, COLORS_39, COLORS_11
import os
from glob import glob
import logging
from util.log import init_logging

init_logging(['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'][4])
logger = logging.getLogger('plot')

PLOTS_FOLDER = f'{OUTPUT_FOLDER}/plots'
os.makedirs(PLOTS_FOLDER, exist_ok=True)

for dataset_name, dataset in DATASETS:
    logger.info(f'> Preparing fake dynamic data for {dataset_name}...')
    for mi, (strategy, name, model_params, strategy_params) in enumerate(MODEL_CONFIGS):
        reader = IntervalResultReader(OUTPUT_FOLDER, dataset_name, strategy, name)
        if len(reader.files) == 0:
            logger.warning('No files seem to be here... Skip')
            continue

        labels, points = reader.get_data()
        # '${dataset}/${model}/${strategy}_${name}_${interval}.tsv',
        output_file = f'{PLOTS_FOLDER}/{dataset_name}_{strategy.__self__.__name__}_{strategy.__name__}_{name}.pdf'

        logger.info(f'Plotting results to {output_file}')

        plot_grid(target_file=output_file, data=points, labels=labels,
                  possible_labels=dataset.data_labels,
                  desc=reader.get_relevant_file_names(),
                  rows=1, cols=len(labels),
                  figsize=(len(labels) * 1.7, 2),
                  point_size=0.1,
                  colours=COLORS_39 if dataset_name == '20news' else COLORS_11)

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
