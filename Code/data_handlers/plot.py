import matplotlib.pylab as plt
import seaborn as sns
import matplotlib
from matplotlib.lines import Line2D
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


COLORS_11 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
             '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99']
COLORS_12 = ["#A5C93D", "#8B006B", "#2000D7", "#538CBA", "#8B006B", "#B33B19",
             "#8B006B", "#8B006B", "#8B006B", "#C38A1F", "#538CBA", "#8B006B"]
COLORS_16 = ["#d7abd4", "#2d74bf", "#9e3d1b", "#3b1b59", "#1b5d2f", "#51bc4c",
             "#ffcb9a", "#768281", "#a0daaa", "#8c7d2b", "#98cc41", "#c52d94",
             "#11337d", "#ff9f2b", "#fea7c1", "#3d672d"]
COLORS_39 = ["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6",
             "#A30059", "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762",
             "#004D43", "#8FB0FF", "#997D87", "#5A0007", "#809693", "#FEFFE6",
             "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A",
             "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA",
             "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018",
             "#0AA6D8", "#013349", "#00846F"]


def plot_grid(target_file, data, labels, desc, rows: int, cols: int, possible_labels,
              figsize: Tuple[Union[int, float], Union[int, float]], point_size=0.2, draw_legend=False,
              colours=None):
    if colours is None:
        colours = COLORS_16
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
        plt.scatter(data_[:, 0], data_[:, 1], s=point_size, c=[colours[int(li)] for li in labels_])
        if desc_ is not None:
            plt.title(desc_)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

        for label in range(len(possible_labels)):
            plt.text(np.mean(data_[labels_ == label, 0]), np.mean(data_[labels_ == label, 1]), label,
                     ha='center', va='center')

    sns.despine(left=True, bottom=True)

    if draw_legend:
        legend_handles = [
            Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colours[li],
                ms=10,
                alpha=1,
                linewidth=0,
                label=l,
                markeredgecolor="k",
            )
            for li, l in enumerate(possible_labels)
        ]
        plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.savefig(target_file)
