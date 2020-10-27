from run_models import OUTPUT_FOLDER, TEMP_FOLDER, MODEL_CONFIGS, DATASETS
from datasets.mnist import MNIST
from datasets.fashion_mnist import FashionMNIST
from datasets.newsgroups import Newsgroups

import numpy as np
from data_handlers.generators import temporal, SamplingReference, accumulate
from data_handlers.disk import ResultWriter, RawDataReaderNP, IntervalDataReader
from evaluation.displacement import calculate_displacement_score
from evaluation.grid import test
from data_handlers.plot import plot_grid, COLORS_39, COLORS_11
import os
from glob import glob
import logging
from util.log import init_logging
from evaluation.isometrics import Measurer as IsometricsMeasurer
import sqlite3 as sql

init_logging(['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'][4])
logger = logging.getLogger('metrics')

METRICS_FOLDER = f'{OUTPUT_FOLDER}/metrics'
os.makedirs(METRICS_FOLDER, exist_ok=True)
DB_FILE = f'{METRICS_FOLDER}/results.db'

RUN = 'drop1'

new_db = not os.path.isfile(DB_FILE)
logger.info(f'Connecting to DB at {DB_FILE}')
conn = sql.connect(DB_FILE)
c = conn.cursor()
if new_db:
    logger.info('DB not initialised yet, creating table.')
    c.execute('CREATE TABLE results( '
              'run TEXT, '
              'dataset TEXT, '
              'model TEXT, '
              'strategy TEXT, '
              'strat_name TEXT, '
              'k INTEGER, '
              'interval INTEGER, '
              'metric TEXT, '
              'value REAL, '
              'space TEXT, '
              'label TEXT, '
              'info TEXT'
              ');')
    conn.commit()


def write_to_db(dataset_, metric_, value_, strategy_, strategy_name, space_=None, label_=None, k_=None, interval_=None,
                info_=None):
    conn.execute(f'INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                 (RUN, dataset_, strategy_.__self__.__name__, strategy_.__name__, strategy_name, k_,
                  interval_, metric_, value_, space_, label_, info_))
    conn.commit()


METRICS = [
    # 'iso.accuracy',
    # 'iso.nmi',
    # 'iso.l-kl',
    'iso.trust',
    'iso.cont'
]

for dataset_name, dataset in DATASETS:
    logger.info(f'> Preparing metric calculations for {dataset_name}...')
    interval_labels, interval_data = RawDataReaderNP(TEMP_FOLDER, dataset_name).get_data()
    for mi, (strategy, name, model_params, strategy_params) in enumerate(MODEL_CONFIGS):
        logger.info(f'> ({mi}/{len(MODEL_CONFIGS)}) Running {strategy.__self__.__name__} ({strategy.__name__}) '
                    f'for {len(interval_data)} intervals...')
        reader = IntervalDataReader(OUTPUT_FOLDER, dataset_name, strategy, name)
        if not reader.check_exist():
            continue
        _, interval_points = reader.get_data()

        for interval, (labels, data_hd, data_ld) in enumerate(zip(interval_labels, interval_data, interval_points)):
            logger.info(f'Calculating metrics for interval {interval} with {len(labels)} items...')
            measurer = IsometricsMeasurer(data_hd, data_ld, labels, k=5)
            for space in ['ld', 'hd']:
                if 'iso.accuracy' in METRICS:
                    logger.info(f'Measure accuracy for {space} and interval {interval}...')
                    acc = measurer.accuracy(space)
                    for k, k_val in enumerate(acc):
                        write_to_db(dataset_name, 'iso.accuracy', k_val, strategy, name,
                                    space_=space, k_=k, interval_=interval)

                if 'iso.nmi' in METRICS:
                    logger.info(f'Measure NMI for {space} and interval {interval}...')
                    nmi = measurer.normalised_mutual_information(space)
                    for k, k_val in enumerate(nmi):
                        write_to_db(dataset_name, 'iso.nmi', k_val, strategy, name,
                                    space_=space, k_=k, interval_=interval)

            if 'iso.l-kl' in METRICS:
                logger.info(f'Measure L-KL...')
                lkl = measurer.local_kullback_leibler()
                write_to_db(dataset_name, 'iso.l-kl', lkl, strategy, name, interval_=interval, info_='sigma=0.001')

            if 'iso.trust' in METRICS:
                logger.info(f'Measure L-KL...')
                trust = measurer.trustworthiness()
                write_to_db(dataset_name, 'iso.trust', trust, strategy, name, interval_=interval)

            if 'iso.cont' in METRICS:
                logger.info(f'Measure L-KL...')
                cont = measurer.continuity()
                write_to_db(dataset_name, 'iso.cont', cont, strategy, name, interval_=interval)
