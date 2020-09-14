import random
from datasets import DataSet
from typing import Union, Dict, List, Tuple
import numpy as np
import time
import enum


class SamplingReference(enum.Enum):
    LABEL_COUNT = enum.auto()
    ITEM_COUNT = enum.auto()


def temporal(dataset: DataSet,
             intervals: Union[int, List[float]] = None,
             label_dist: List[Dict[int, float]] = None,
             target_size: Union[int, float] = None,
             auto_fill: bool = True,
             sampling_reference: SamplingReference = SamplingReference.ITEM_COUNT,
             rand_seed: str = '43') -> Tuple[List[List[np.array]], List[List[int]]]:
    """
    TODO rewrite, so it's not in-memory

    If some distribution information is not provided, equal distribution is assumed

    :param dataset: The dataset to sample from
    :param intervals: Either the number of intervals or list of floats of length intervals and distribution
                      If None, it's derived from label_dist
    :param label_dist: Iterable of length intervals, with dict describing label distribution in each interval
    :param target_size: Size of the final dataset
    :param auto_fill: Automatically expand on labels not defined in label_dist
    :param sampling_reference: Bases for partial data expansion
    :param rand_seed: Seed for random shuffling
    :return:
    """
    random.seed(rand_seed)

    num_items = len(dataset)
    num_labels = len(dataset.data_labels)

    if intervals is None:
        intervals = len(label_dist)
    if type(intervals) is int:
        intervals = [1 / intervals for _ in range(intervals)]

    if target_size is None:
        target_size = num_items
    elif type(target_size) is float:
        target_size = target_size * num_items

    if label_dist is None:
        label_dist = [{label: 1 / num_labels} for _ in intervals for label in dataset.data_labels]

    data = []
    labels = []

    idxs = [[] for _ in range(num_labels)]
    for i, label in enumerate(dataset.get_labels()):
        idxs[label].append(i)
    for iidxs in idxs:
        random.shuffle(iidxs)

    print(f'Sampling data with\n'
          f'  - intervals: [{", ".join(f"{i:.3f}" for i in intervals)}]\n'
          f'  - target size: {target_size}\n'
          f'  - actual labels: {" | ".join(f"label {i} ({len(iidxs)})" for i, iidxs in enumerate(idxs))}\n'
          f'  - requested label distribution: {label_dist}')

    current_idxs = [0] * num_labels

    for interval_num, (interval_size, label_distribution) in enumerate(zip(intervals, label_dist)):
        data_interval = []
        labels_interval = []
        if auto_fill and len(label_distribution) != num_labels:
            preliminary_total = sum(ld if type(ld) is float else ld / len(idxs[l])
                                    for l, ld in label_distribution.items())
            default_dist = (1 - preliminary_total) / (num_labels - len(label_distribution))
            label_distribution = {label: default_dist if label not in label_distribution else label_distribution[label]
                                  for label in dataset.data_labels}
        for label, distribution in label_distribution.items():
            print(f'interval {interval_num + 1}, label {label} (current idx: {current_idxs[label]}):', end=' ')
            if type(distribution) is int:
                interval_label_size_abs = distribution
                print(f'fixed abs = {distribution} items')
            else:
                if sampling_reference is SamplingReference.ITEM_COUNT:
                    abs_factor = target_size
                else:
                    abs_factor = len(idxs[label]) * (target_size / num_items) * num_labels
                interval_label_size_abs = int(round(distribution * interval_size * abs_factor))
                print(f'calculated abs({distribution:.3f}*{interval_size:.3f}*{abs_factor:.3f}) '
                      f'= {interval_label_size_abs} items')

            if len(idxs[label]) < current_idxs[label] + interval_label_size_abs:
                print(f' > WARN: not enough data for label {label} at interval {interval_num}'
                      f' (diff: {len(idxs[label]) - current_idxs[label] - interval_label_size_abs})')
            labels_interval += [label] * interval_label_size_abs
            data_interval += [dataset.get_data()[idx] for idx in
                              idxs[label][current_idxs[label]:current_idxs[label] + interval_label_size_abs]]
            current_idxs[label] += interval_label_size_abs
        print(f'== Interval {interval_num + 1} has {len(data_interval)} items in total.')
        data.append(data_interval)
        labels.append(labels_interval)
    return data, labels


def accumulate(data, labels, generate=True):
    accumulator_data = []
    accumulator_labels = []
    ret = []
    for interval_data, interval_labels in zip(data, labels):
        accumulator_data += interval_data
        accumulator_labels += interval_labels
        if generate:
            yield accumulator_data, accumulator_labels
        else:
            ret.append((accumulator_data, accumulator_labels))
    if not generate:
        return ret
