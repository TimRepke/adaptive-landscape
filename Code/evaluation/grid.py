import os
import numpy as np
from typing import List, Tuple
import logging
from evaluation import _read_file
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

logger = logging.getLogger('displacement')


def _compute_grid_assignments(arr, grid_size):
    arr = arr[:, [0, 1]]
    grid_size = np.array(grid_size)
    mi = arr.min(axis=0)
    ma = arr.max(axis=0)

    offset = np.abs(mi) * (-1) * np.sign(mi)
    cell_size = (np.abs(mi) + np.abs(ma)) / grid_size
    gs = np.array([grid_size[0], 1])

    return np.array([np.sum((np.minimum(pos + offset, grid_size - 1) // cell_size) * gs) for pos in arr])


def calculate_cell_spread(arr, threshold, grid_size):
    assignments = _compute_grid_assignments(arr, grid_size)
    cell_sizes = Counter(assignments)
    labels = arr[:, 2]
    label_spread = {}
    for label in set(labels):
        cells = Counter(assignments[labels == label])
        purities = np.array([cell_c / cell_sizes[cell_i] for cell_i, cell_c in cells.items()])
        label_spread[label] = sum(purities < threshold) / (grid_size[0] * grid_size[1])
    return label_spread


def calculate_label_purities(arr, threshold=0.9, grid_size=None):
    if not grid_size:
        grid_size = (10, 10)

    assignments = _compute_grid_assignments(arr, grid_size)
    cell_sizes = Counter(assignments)
    labels = arr[:, 2]
    label_purities = {}
    for label in set(labels):
        cells = Counter(assignments[labels == label])
        purities = np.array([cell_c / cell_sizes[cell_i] for cell_i, cell_c in cells.items()])
        label_purities[label] = sum(purities > threshold) / len(cells)
    return label_purities


def label_purity_parameter_plot(arr, thresholds: List[float] = None, grids=None):
    if not grids:
        grids = list(range(10, 100, 5))
    if not thresholds:
        thresholds = [0.6, 0.8, 0.9, 1.0]

    fig, ax = plt.subplots()
    for threshold in thresholds:
        points = [np.mean(list(
            calculate_label_purities(arr, threshold=threshold, grid_size=(grid_i, grid_i)).values()))
            for grid_i in grids
        ]
        ax.plot(grids, points, label=f'th={threshold:.2f}')
    ax.legend()
    plt.show()


def spread_purity_plot(base_files, threshold_spread: float = None, threshold_purity: float = None,
                       grid_size: Tuple[int, int] = None):
    if not grid_size:
        grid_size = (30, 30)
    if not threshold_spread:
        threshold_spread = 0.1
    if not threshold_purity:
        threshold_purity = 0.9

    plots_spread = defaultdict(list)
    plots_purity = defaultdict(list)

    for plots, threshold, func in [
        (plots_purity, threshold_purity, calculate_label_purities),
        (plots_spread, threshold_spread, calculate_cell_spread)
    ]:
        for base_file in base_files:
            arr = _read_file(base_file)
            [plots[label].append(val)
             for label, val in func(arr, threshold, grid_size).items()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(os.path.basename(base_files[0]))
    ax1.set_title('Spread')
    ax2.set_title('Purity')

    for label, plot in plots_spread.items():
        ax1.plot(range(len(base_files)), plot, label=label, marker='.', linestyle=':')
    for label, plot in plots_purity.items():
        ax2.plot(range(len(base_files)), plot, label=label, marker='.', linestyle=':')

    # ax1.legend()
    # ax2.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()


def calculate_displacement_score(base_files):
    base_files.sort()
    prev = _read_file(base_files[0])
    for file in base_files[1:]:
        curr = _read_file(file)
        logger.debug(f'Loaded file: {file}')
        displacement = np.linalg.norm((prev[:, [0, 1]] / np.linalg.norm(prev[:, [0, 1]])) -
                                      (curr[:len(prev), [0, 1]] / np.linalg.norm(curr[:, [0, 1]])), axis=1)
        logger.info(f'Number of compared points: {displacement.shape[0]}')
        logger.debug(displacement)

        logger.info(f'Displacement per digit: ' +
                    str({digit: np.mean(displacement[prev[:, 2] == digit]) for digit in range(10)}))

        logger.info(f'Total displacement: {displacement.mean():.7f}')
        prev = curr


def test(base_files):
    base_files.sort()
    prev = _read_file(base_files[0])
    # label_purity_plot(prev, [0.7, 0.8, 0.9], grids=list(range(10, 100, 5)))
    spread_purity_plot(base_files, 0.1, 0.8, (50, 50))
