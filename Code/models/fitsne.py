import numpy as np
from models import Model
from externals.FItSNE.fast_tsne import fast_tsne
from dataclasses import dataclass, asdict
from typing import Union, Callable, Iterable, Optional
from numpy.random import RandomState
import logging
import os

logger = logging.getLogger('fitsne')


@dataclass
class FItSNEParams:
    # Set to 0 for exact t-SNE. If non-zero, then the code will use either
    # Barnes Hut or FIt-SNE based on `nbody_algo`.  If Barnes Hut, then theta
    # determins the accuracy of BH approximation. Default 0.5.
    theta: float = 0.5
    # Perplexity is used to determine the bandwidth of the Gaussian kernel
    # in the input space.  Default 30.
    perplexity: int = 30
    # Number of embedding dimensions. Default 2. FIt-SNE supports only 1 or 2
    # dimensions.
    map_dims: int = 2
    # Number of gradient descent iterations. Default 750.
    max_iter: int = 750
    # stop_early_exag_iter: int
    # When to switch off early exaggeration. Default 250.
    stop_early_exag_iter: int = 250
    # K: int
    # The number of nearest neighbours to use when using fixed sigma instead
    # of perplexity calibration. Set to -1 when perplexity is used. Default
    # is -1.
    K: int = -1
    # sigma: boolean
    # The standard deviation of the Gaussian kernel to be used for all points
    # instead of choosing it adaptively via perplexity. Set to -1 to use
    # perplexity. Default is -1.
    sigma: int = -1
    # {'Barnes-Hut', 'FFT'}
    # If theta is nonzero, this determines whether to use FIt-SNE (default) or
    # Barnes-Hut approximation.
    nbody_algo: str = "FFT"
    # knn_algo: {'vp-tree', 'annoy'}
    # Use exact nearest neighbours with VP trees (as in BH t-SNE) or
    # approximate nearest neighbors with Annoy. Default is 'annoy'.
    knn_algo: str = "annoy"
    # mom_switch_iter: int
    # Iteration number to switch from momentum to final_momentum. Default 250.
    mom_switch_iter: int = 250
    # momentum: double
    # Initial value of momentum. Default 0.5.
    momentum: float = 0.5
    # final_momentum: double
    # The value of momentum to use later in the optimisation. Default 0.8.
    final_momentum: float = 0.8
    # learning_rate: double or 'auto'
    # Learning rate. Default 'auto'; it sets learning rate to
    # N/early_exag_coeff where N is the sample size, or to 200 if
    # N/early_exag_coeff < 200.
    learning_rate: Union[str, float] = "auto"
    # early_exag_coeff: double
    # Coefficient for early exaggeration. Default 12.
    early_exag_coeff: int = 12
    # no_mometum_during_exag: boolean
    # Whether to switch off momentum during the early exaggeration phase (can
    # be useful for experiments with large exaggeration coefficients). Default
    # is False.
    no_momentum_during_exag: bool = False
    # n_trees: int
    # When using Annoy, the number of search trees to use. Default is 50.
    n_trees: int = 50
    # search_k: int
    # When using Annoy, the number of nodes to inspect during search. Default
    # is 3*perplexity*n_trees (or K*n_trees when using fixed sigma).
    search_k: Optional[int] = None
    # start_late_exag_iter: int or 'auto'
    # When to start late exaggeration. Default 'auto'; it sets
    # start_late_exag_iter to -1 meaning that late exaggeration is not used,
    # unless late_exag_coeff>0. In that case start_late_exag_iter is set to
    # stop_early_exag_iter.
    start_late_exag_iter: Union[int, str] = "auto"
    # late_exag_coeff: double
    # Coefficient for late exaggeration. Set to -1 in order not to use late
    # exaggeration. Default -1.
    late_exag_coeff: int = -1
    # nterms: int
    # If using FIt-SNE, this is the number of interpolation points per
    # sub-interval
    nterms: int = 3
    # intervals_per_integer: double
    # See min_num_intervals
    intervals_per_integer: int = 1
    # min_num_intervals: int
    # The interpolation grid is chosen on each step of the gradient descent.
    # If Y is the current embedding, let maxloc = ceiling(max(Y.flatten)) and
    # minloc = floor(min(Y.flatten)), i.e. the points are contained in a
    # [minloc, maxloc]^no_dims box. The number of intervals in each
    # dimension is either min_num_intervals or
    # ceiling((maxloc-minloc)/intervals_per_integer), whichever is larger.
    # min_num_intervals must be a positive integer and intervals_per_integer
    # must be positive real value. Defaults: min_num_intervals=50,
    # intervals_per_integer = 1.
    min_num_intervals: int = 50
    # seed: int
    # Seed for random initialisation. Use -1 to initialise random number
    # generator with current time. Default -1.
    seed: int = -1
    # initialization: 'random', 'pca', or numpy array
    # N x no_dims array to intialize the solution. Default: 'pca'.
    initialization: Union[str, np.array] = "pca"
    # load_affinities: {'load', 'save', None}
    # If 'save', input similarities (p_ij) are saved into a file. If 'load',
    # they are loaded from a file and not recomputed. If None, they are not
    # saved and not loaded. Default is None.
    load_affinities: Optional[str] = None
    # perplexity_list: list
    # A list of perplexities to used as a perplexity combination. Input
    # affinities are computed with each perplexity on the list and then
    # averaged. Default is None.
    perplexity_list: Optional[list] = None
    # df: double
    # Controls the degree of freedom of t-distribution. Must be positive. The
    # actual degree of freedom is 2*df-1. The standard t-SNE choice of 1
    # degree of freedom corresponds to df=1. Large df approximates Gaussian
    # kernel. df<1 corresponds to heavier tails, which can often resolve
    # substructure in the embedding. See Kobak et al. (2019) for details.
    # Default is 1.0.
    df: float = 1
    # return_loss: boolean
    # If True, the function returns the loss values computed during
    # optimisation together with the final embedding. If False, only the
    # embedding is returned. Default is False.
    return_loss: bool = False
    # nthreads: int
    # Number of threads to use. Default is -1, i.e. use all available threads.
    nthreads: int = -1
    # max_step_norm: double or 'none' (default: 5)
    # Maximum distance that a point is allowed to move on one iteration.
    # Larger steps are clipped to this value. This prevents possible
    # instabilities during gradient descent. Set to 'none' to switch it off.
    max_step_norm: int = 5


@dataclass
class FItSNEStrategyParams:
    k: int = 5


class FItSNEModel(Model):
    @classmethod
    def fit_data(cls, data=None, params=None):
        """Run t-SNE. This implementation supports exact t-SNE, Barnes-Hut t-SNE
        and FFT-accelerated interpolation-based t-SNE (FIt-SNE). This is a Python
        wrapper to a C++ executable.

        Returns
        -------
        Y: numpy array
            The embedding.
        loss: numpy array
            Loss values computed during optimisation. Only returned if return_loss is True.
        """
        if params is None:
            params = FItSNEParams()
        if data is not None:
            data = np.array(data, dtype=np.float)
        else:
            raise AssertionError('Needs either data or input_file parameter!')
        points = fast_tsne(data, **asdict(params))
        cls._cleanup()
        return points

    @staticmethod
    def _cleanup():
        os.remove('result.dat')
        os.remove('data.dat')

    @classmethod
    def _strategy_flex(cls, init_func: Callable, data: Iterable, params: FItSNEParams,
                       strategy_params: FItSNEStrategyParams):
        """"This strategy uses data from the previous layout
            as initialisation for the next.
        """
        if strategy_params is None:
            strategy_params = FItSNEStrategyParams()

        # containers for the previous high- and low-dimensional data
        prev_2d = None
        prev_hd = None
        for interval, (interval_data, interval_labels) in enumerate(data):
            logger.info(f'> Running FItSNE ({init_func.__name__}) for interval {interval} '
                        f'with {len(interval_data)} data points.')

            if prev_hd is not None:
                params.initialization = init_func(prev_2d, prev_hd, interval_data, strategy_params.k)

            prev_2d = fast_tsne(interval_data, **asdict(params))
            prev_hd = interval_data

            yield interval_labels, prev_2d
        cls._cleanup()

    @classmethod
    def strategy_flex_mean(cls, data: Iterable, params: FItSNEParams, strategy_params: FItSNEStrategyParams):
        """"This strategy uses data from the previous layout
            as initialisation for the next.
            New points are mean of previous kNN
        """
        yield from cls._strategy_flex(cls.get_init_mean, data, params, strategy_params)

    @classmethod
    def strategy_flex_mean_weighted(cls, data: Iterable, params: FItSNEParams, strategy_params: FItSNEStrategyParams):
        """"This strategy uses data from the previous layout
            as initialisation for the next.
            New points are weighted (by distance) mean of previous kNN
        """
        yield from cls._strategy_flex(cls.get_init_mean_weighted, data, params, strategy_params)

    @classmethod
    def strategy_flex_median(cls, data: Iterable, params: FItSNEParams, strategy_params: FItSNEStrategyParams):
        """"This strategy uses data from the previous layout
            as initialisation for the next.
            New points are median of previous kNN
        """
        yield from cls._strategy_flex(cls.get_init_median, data, params, strategy_params)
