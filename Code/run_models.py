from datasets.mnist import MNIST
from datasets.fashion_mnist import FashionMNIST
from datasets.newsgroups import Newsgroups

import numpy as np
from data_handlers.generators import temporal, SamplingReference, accumulate
from data_handlers.disk import ResultWriter
from models.large_vis import LargeVisModel, LargeVisParams
from models.fitsne import FItSNEModel, FItSNEParams, FItSNEStrategyParams
from models.bhtsne import BHtSNEModel
from models.open_tsne import OpenTSNEModel, OpenTSNEParams
from models.open_tsne import PerplexityParams, SigmaParams, UniformParams
from models.umap import UMAPModel, UMAPParams
from models.param_umap import ParametricUMAPModel, ParamUMAPParams
from models.thesne import DynTSNEModel, DynTSNEParams
from evaluation.displacement import calculate_displacement_score
from evaluation.grid import test
from data_handlers.plot import plot_grid
import os
from glob import glob
import logging
from util.log import init_logging

init_logging(['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'][4])
logger = logging.getLogger('main')

TEMP_FOLDER = '../TempData'
OUTPUT_FOLDER = '../OutputData'

logger.info('> Loading datasets...')
DATASETS = [
    ('20news', Newsgroups('../RawData/20news', f'{TEMP_FOLDER}/20news/parsed')),
    # ('mnist', MNIST('../RawData/MNIST')),
    # ('fashion_mnist', FashionMNIST('../RawData/FashionMNIST'))
]

# LABEL_DIST = [{3: 0.001}, {3: 0.01}, {3: 0.1}]
# LABEL_DIST = [{3: 0.001}, {3: 0.005}, {3: 0.01}, {3: 0.05}, {3: 0.1}]
# LABEL_DIST = [
#    {l: 800 if l != 1 else 1 for l in range(20)},
#    {1: 100},
#    {1: 300},
#    {1: 600}
# ]
LABEL_DIST = [
    {l: 200 if l != 1 else 1 for l in range(20)},
    {1: 10},
    {1: 50},
    {1: 140}
]
SAMPLING_REF = SamplingReference.LABEL_COUNT
TARGET_SIZE = 20000

MODEL_CONFIGS = [
    (FItSNEModel.strategy_static, 'df05', FItSNEParams(seed=2020, df=0.5), None),
    (FItSNEModel.strategy_static, 'df1', FItSNEParams(seed=2020, df=1.), None),
    (FItSNEModel.strategy_static, 'df100', FItSNEParams(seed=2020, df=100.), None),
    (FItSNEModel.strategy_flex_mean, 'df1_k5', FItSNEParams(seed=2020, df=1.), FItSNEStrategyParams(k=5)),
    (FItSNEModel.strategy_flex_mean, 'df05_k5', FItSNEParams(seed=2020, df=.5), FItSNEStrategyParams(k=5)),
    (FItSNEModel.strategy_flex_mean_weighted, 'df1_k5', FItSNEParams(seed=2020, df=1.), FItSNEStrategyParams(k=5)),
    (FItSNEModel.strategy_flex_median, 'df1_k5', FItSNEParams(seed=2020, df=1.), FItSNEStrategyParams(k=5)),
    (UMAPModel.strategy_static, 'cos', UMAPParams(metric='cosine'), None),
    (UMAPModel.strategy_fix, 'cos', UMAPParams(metric='cosine'), None),
    (UMAPModel.strategy_flex, 'cos', UMAPParams(metric='cosine'), None),
    (UMAPModel.strategy_semi_fix, 'cos', UMAPParams(metric='cosine'), None),
    (OpenTSNEModel.strategy_static, 'pca', OpenTSNEParams(initialization='pca'), None),
    (OpenTSNEModel.strategy_static, 'spec', OpenTSNEParams(initialization='spectral'), None),
    (OpenTSNEModel.strategy_perplexity, 'spec', OpenTSNEParams(initialization='spectral'),
     PerplexityParams(method='hnswlib')),
    (OpenTSNEModel.strategy_sigma, 'defaults', OpenTSNEParams(), SigmaParams(method='hnswlib')),
    (OpenTSNEModel.strategy_uniform, 'defaults', OpenTSNEParams(), UniformParams(method='hnswlib')),
    (LargeVisModel.strategy_static, 'defaults', LargeVisParams(), None),
    (ParametricUMAPModel.strategy_static, 'default', ParamUMAPParams(n_training_epochs=5), None),
    (ParametricUMAPModel.strategy_fix, 'default', ParamUMAPParams(n_training_epochs=5), None),
    (ParametricUMAPModel.strategy_semi_fix, 'default', ParamUMAPParams(n_training_epochs=5), None),
    (ParametricUMAPModel.strategy_flex, 'default', ParamUMAPParams(n_training_epochs=5), None),
    # (DynTSNEModel.strategy_native, 'default', DynTSNEParams(random_state=2020), None) # FIXME?
    # TODO isomap
    # TODO TopoAE
    # TODO MLDL
]

if __name__ == '__main__':
    for dataset_name, dataset in DATASETS:
        logger.info(f'> Preparing fake dynamic data for {dataset_name}...')
        dataset.load()
        temporal_data, temporal_labels = temporal(dataset,
                                                  label_dist=LABEL_DIST,
                                                  target_size=TARGET_SIZE,
                                                  sampling_reference=SAMPLING_REF,
                                                  auto_fill=False)
        intervals = list(accumulate(temporal_data, temporal_labels))
        for mi, (strategy, name, model_params, strategy_params) in enumerate(MODEL_CONFIGS):
            logger.info(f'> ({mi}/{len(MODEL_CONFIGS)}) Running {strategy.__self__.__name__} ({strategy.__name__}) '
                        f'for {len(intervals)} intervals...')
            logger.debug(f'  - Model params: {model_params}')
            logger.debug(f'  - Strategy params: {strategy_params}')

            try:
                result_writer = ResultWriter(OUTPUT_FOLDER, dataset_name, strategy, name)
                for interval, (interval_labels, positions) in \
                        enumerate(strategy(intervals, model_params, strategy_params)):
                    result_writer.store_result(interval, interval_labels, positions)
            except FileExistsError as e:
                logger.warning(getattr(e, 'message', repr(e)))
