import matplotlib.pylab as plt
import seaborn as sns
import matplotlib
from typing import Tuple, Union
from itertools import zip_longest
import numpy as np


def sns_styleset():
    sns.set_context('paper')
    sns.set_style('ticks')
    matplotlib.rcParams['axes.linewidth'] = .75
    matplotlib.rcParams['xtick.major.width'] = .75
    matplotlib.rcParams['ytick.major.width'] = .75
    matplotlib.rcParams['xtick.major.size'] = 3
    matplotlib.rcParams['ytick.major.size'] = 3
    matplotlib.rcParams['xtick.minor.size'] = 2
    matplotlib.rcParams['ytick.minor.size'] = 2
    matplotlib.rcParams['font.size'] = 7
    matplotlib.rcParams['axes.titlesize'] = 7
    matplotlib.rcParams['axes.labelsize'] = 7
    matplotlib.rcParams['legend.fontsize'] = 7
    matplotlib.rcParams['xtick.labelsize'] = 7
    matplotlib.rcParams['ytick.labelsize'] = 7


col = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
       '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99']


def plot_grid(target_file, data, labels, desc, rows: int, cols: int,
              figsize: Tuple[Union[int, float], Union[int, float]], point_size=0.2):
    # figsize: Width, height in inches.
    if desc is None:
        desc = []
    sns_styleset()
    fig = plt.figure(figsize=figsize)
    cell_height = 1 / rows  # figsize[1] / rows
    cell_width = 1 / cols  # figsize[0] / cols
    ax = []
    for ri in range(rows):
        for ci in range(cols):
            # [left, bottom, width, height]`
            a = fig.add_axes([ci * cell_width, ri * cell_height, cell_width, cell_height],
                             label=f'{ri}_{ci}', aspect='equal', adjustable='box')
            ax.append(a)

    for i, (data_, labels_, desc_) in enumerate(zip_longest(data, labels, desc)):
        plt.sca(ax[i])
        plt.scatter(data_[:, 0], data_[:, 1], s=point_size, c=[col[int(li)] for li in labels_])
        if desc_ is not None:
            plt.title(desc_)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

        for digit in range(10):
            plt.text(np.mean(data_[labels_ == digit, 0]), np.mean(data_[labels_ == digit, 1]), digit,
                     ha='center', va='center')

    sns.despine(left=True, bottom=True)
    plt.savefig(target_file)
