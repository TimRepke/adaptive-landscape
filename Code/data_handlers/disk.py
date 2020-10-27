from string import Template
import os
import logging
from glob import glob
import numpy as np
from typing import Literal

logger = logging.getLogger('disk')


class ResultWriter:
    def __init__(self, target_folder, dataset, strategy, name,
                 base_template='${dataset}/${model}/${strategy}_${name}_${interval}.tsv',
                 info=None, check_override=True, force_override=False):
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
                if force_override:
                    logger.warning('File exists but I don\'t care!')
                else:
                    logger.warning('File exists, stopping here.')
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


class RawDataWriter(ResultWriter):
    def __init__(self, target_folder, dataset,
                 base_template='${dataset}/intervals/interval_${interval}.tsv', info=None, check_override=True,
                 force_override=False):
        if info is None:
            info = {'dataset': dataset}

        super().__init__(target_folder=target_folder, dataset=dataset, strategy=None, name=None,
                         base_template=base_template, info=info, check_override=check_override,
                         force_override=force_override)


class IntervalDataReader:
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
        self._glob_pattern = os.path.join(self.target_folder, Template(self.base_template).substitute(self.info))
        self.files = sorted(glob(self._glob_pattern))

    def check_exist(self):
        if len(self.files) > 0:
            logger.info(f'Found {len(self.files)} for pattern {self._glob_pattern}')
            return True
        logger.warning(f'Found {len(self.files)} for pattern {self._glob_pattern}')
        return False

    def get_relevant_file_names(self):
        return self.files

    @staticmethod
    def s_generate_intervals(files):
        for file_name in files:
            logger.debug(f'Reading {file_name}...')
            with open(file_name, 'r') as f:
                tmp = [
                    [float(li) for li in line.strip().split()]
                    for line in f
                ]
                # labels, points
                yield np.array([int(ti[-1]) for ti in tmp]), \
                      np.array([ti[:-1] for ti in tmp])

    def generate_intervals(self):
        yield from self.s_generate_intervals(self.files)

    def get_data(self):
        labels = []
        points = []
        for l, p in self.generate_intervals():
            labels.append(l)
            points.append(p)
        return labels, points


class RawDataReaderNP:
    def __init__(self, target_folder, dataset,
                 base_template='${dataset}/intervals/interval_${interval}.${filetype}', info=None):
        if info is None:
            info = {'dataset': dataset, 'interval': '*', 'filetype': 'npy'}

        glob_pattern = os.path.join(target_folder, Template(base_template).substitute(info))
        self.files = sorted(glob(glob_pattern))

        if len(self.files) == 0:
            logger.info('Numpy translation doesn\'t exist yet. Converting now.')
            info['filetype'] = 'tsv'
            glob_pattern = os.path.join(target_folder, Template(base_template).substitute(info))
            files = sorted(glob(glob_pattern))

            for filename, (labels, points) in zip(files, IntervalDataReader.s_generate_intervals(files)):
                np_filename = filename.replace('.tsv', '.npy')
                with open(np_filename, 'wb') as f:
                    np.save(f, points)
                    np.save(f, labels)

            info['filetype'] = 'npy'
            glob_pattern = os.path.join(target_folder, Template(base_template).substitute(info))
            self.files = sorted(glob(glob_pattern))

    def generate_intervals(self):
        for file_name in self.files:
            logger.debug(f'Reading {file_name}...')
            with open(file_name, 'rb') as f:
                points = np.load(f)
                labels = np.load(f)
                yield labels, points

    def get_data(self):
        labels = []
        points = []
        for l, p in self.generate_intervals():
            labels.append(l)
            points.append(p)
        return labels, points

