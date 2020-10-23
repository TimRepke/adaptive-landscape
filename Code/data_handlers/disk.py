from string import Template
import os
import logging
from glob import glob
import numpy as np

logger = logging.getLogger('disk')


class ResultWriter:
    def __init__(self, target_folder, dataset, strategy, name,
                 base_template='${dataset}/${model}/${strategy}_${name}_${interval}.tsv',
                 info=None,
                 check_override=True):
        self.base_template = base_template
        self.target_folder = target_folder
        self.info = info
        if self.info is None:
            self.info = {
                'dataset': dataset,
                'name': name,
                'interval': 0,
                'model': strategy.__self__.__name__,
                'strategy': strategy.__name__
            }

        file_name = self._get_filename(0)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        if check_override:
            logger.info(f'Checking if results already exist for this experiment in {file_name}')
            if os.path.isfile(file_name):
                logger.warning('File exists!')
                raise FileExistsError('Please make sure you actually want to override existing results.')

    def _get_filename(self, interval):
        self.info['interval'] = interval
        return os.path.join(self.target_folder, Template(self.base_template).substitute(self.info))

    def store_result(self, interval, labels, vectors):
        file_name = self._get_filename(interval)
        logger.info(f'> Writing output to {file_name}...')
        with open(file_name, 'w') as out:
            [
                out.write(' '.join(f'{dim:.5f}' for dim in vector) + f' {label}\n')
                for vector, label in zip(vectors, labels)
            ]


class IntervalResultReader:
    def __init__(self, target_folder, dataset, strategy, name,
                 base_template='${dataset}/${model}/${strategy}_${name}_${interval}.tsv',
                 info=None):
        self.base_template = base_template
        self.target_folder = target_folder
        self.info = info
        if self.info is None:
            self.info = {
                'dataset': dataset,
                'name': name,
                'interval': '*',
                'model': strategy.__self__.__name__,
                'strategy': strategy.__name__
            }
        self.files = sorted(glob(os.path.join(self.target_folder, Template(self.base_template).substitute(self.info))))

    def get_relevant_file_names(self):
        return self.files

    def generate_intervals(self):
        for file_name in self.files:
            logger.debug(f'Reading {file_name}...')
            with open(file_name, 'r') as f:
                tmp = [
                    [float(li) for li in line.strip().split()]
                    for line in f
                ]
                # labels, points
                yield np.array([int(ti[-1]) for ti in tmp]), \
                      np.array([[ti[0], ti[1]] for ti in tmp])

    def get_data(self):
        labels = []
        points = []
        for l, p in self.generate_intervals():
            labels.append(l)
            points.append(p)
        return labels, points
