import numpy as np
from models import Model
import enum

from externals.thesne.model.dynamic_tsne import dynamic_tsne
from dataclasses import dataclass, asdict
from typing import Union, Callable, Iterable, Optional, List, Type, Literal
import scipy.sparse
from numpy.random import RandomState
import logging

logger = logging.getLogger('DynTSNE')


@dataclass
class DynTSNEParams:
    # Xs : list of array-likes, each with shape (n_observations, n_features), \
    # or (n_observations, n_observations) if `metric` == 'precomputed'.
    # List of matrices containing the observations (one per row). If `metric`
    # is 'precomputed', list of pairwise dissimilarity (distance) matrices.
    # Each row in `Xs[t + 1]` should correspond to the same row in `Xs[t]`,
    # for every time step t > 1.
    Xs: Optional[List[np.ndarray]] = None
    # perplexity : float, optional (default = 30)
    # Target perplexity for binary search for sigmas.
    perplexity: float = 30
    # Ys : list of array-likes, each with shape (n_observations, output_dims), \
    # optional (default = None)
    # List of matrices containing the starting positions for each point at
    # each time step.
    Ys: Optional[List[np.ndarray]] = None
    # output_dims : int, optional (default = 2)
    # Target dimension.
    output_dims: int = 2
    # n_epochs : int, optional (default = 1000)
    # Number of gradient descent iterations.
    n_epochs: int = 1000
    # initial_lr : float, optional (default = 2400)
    # The initial learning rate for gradient descent.
    initial_lr: float = 2400
    # final_lr : float, optional (default = 200)
    # The final learning rate for gradient descent.
    final_lr: float = 200
    # lr_switch : int, optional (default = 250)
    # Iteration in which the learning rate changes from initial to final.
    # This option effectively subsumes early exaggeration.
    lr_switch: int = 250
    # init_stdev : float, optional (default = 1e-4)
    # Standard deviation for a Gaussian distribution with zero mean from
    # which the initial coordinates are sampled.
    init_stdev: float = 1e-4
    # sigma_iters : int, optional (default = 50)
    # Number of binary search iterations for target perplexity.
    sigma_iters: int = 50
    # initial_momentum : float, optional (default = 0.5)
    # The initial momentum for gradient descent.
    initial_momentum: float = 0.5
    # final_momentum : float, optional (default = 0.8)
    # The final momentum for gradient descent.
    final_momentum: float = 0.8
    # momentum_switch : int, optional (default = 250)
    # Iteration in which the momentum changes from initial to final.
    momentum_switch: int = 250
    # lmbda : float, optional (default = 0.0)
    # Movement penalty hyperparameter. Controls how much each point is
    # penalized for moving across time steps.
    lmbda: float = 0.0
    # metric : 'euclidean' or 'precomputed', optional (default = 'euclidean')
    # Indicates whether `X[t]` is composed of observations ('euclidean')
    # or distances ('precomputed'), for all t.
    metric: Literal['euclidean', 'precomputed'] = 'euclidean'
    # random_state : int or np.RandomState, optional (default = None)
    # Integer seed or np.RandomState object used to initialize the
    # position of each point. Defaults to a random seed.
    random_state: Optional[Union[RandomState, int]] = None
    # verbose : bool (default = 1)
    # Indicates whether progress information should be sent to standard
    # output.
    verbose: bool = 1


class DynTSNEModel(Model):
    @classmethod
    def fit_data(cls, data=None, params: DynTSNEParams = None):
        raise NotImplementedError('Please use strategy_native instead.')

    @classmethod
    def strategy_native(cls, data: Iterable, params: DynTSNEParams, strategy_params):
        logger.info('Unpacking data...')
        data_hd = [d[0] for d in data]
        labels = [d[1] for d in data]

        logger.info('Running dynamic tSNE...')
        params.Xs = data_hd
        points = dynamic_tsne(**asdict(params))

        logger.info('Done. Yielding data now...')
        for i_labels, i_points in zip(labels, points):
            yield i_labels, np.array(i_points)
