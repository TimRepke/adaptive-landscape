from run_models import OUTPUT_FOLDER, MODEL_CONFIGS, DATASETS
from datasets.mnist import MNIST
from datasets.fashion_mnist import FashionMNIST
from datasets.newsgroups import Newsgroups

import numpy as np
from data_handlers.generators import temporal, SamplingReference, accumulate
from data_handlers.disk import ResultWriter
from evaluation.displacement import calculate_displacement_score
from evaluation.grid import test
from data_handlers.plot import plot_grid
import os
from glob import glob
import logging
from util.log import init_logging

init_logging(['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'][4])
logger = logging.getLogger('metrics')

for dataset_name, _ in DATASETS:
    for mi, (model, _) in enumerate(MODEL_CONFIGS):
        logger.info(f'Running evaluations for {dataset_name} and {model} ({mi})...')
        eval_files = glob(f'{OUTPUT_FOLDER}/{dataset_name}_{model.__name__}{mi}*.tsv')
        test(eval_files)
        # calculate_displacement_score(eval_files)
