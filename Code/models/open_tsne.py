import numpy as np
from models import Model
import enum

from openTSNE import TSNE, TSNEEmbedding, PartialTSNEEmbedding
from openTSNE.affinity import Affinities, PerplexityBasedNN, FixedSigmaNN, Uniform
from openTSNE.callbacks import ErrorLogger
from openTSNE import initialization
from dataclasses import dataclass, asdict
from typing import Union, Callable, Iterable, Optional, List, Type, Literal
import scipy.sparse
from numpy.random import RandomState
import logging


logger = logging.getLogger('openTSNE')


@dataclass
class OpenTSNEParams:
    # n_components: int
    # The dimension of the embedding space. This deafults to 2 for easy
    # visualization, but sometimes 1 is used for t-SNE heatmaps. t-SNE is
    # not designed to embed into higher dimension and please note that
    # acceleration schemes break down and are not fully implemented.
    n_components = 2
    # perplexity: float
    # Perplexity can be thought of as the continuous :math:`k` number of
    # nearest neighbors, for which t-SNE will attempt to preserve distances.
    perplexity: float = 30
    # learning_rate: Union[str, float]
    # The learning rate for t-SNE optimization. When ``learning_rate="auto"``
    # the appropriate learning rate is selected according to max(200, N / 12),
    # as determined in Belkina et al. "Automated optimized parameters for
    # T-distributed stochastic neighbor embedding improve visualization and
    # analysis of large datasets", 2019.
    learning_rate: Union[str, float] = "auto"
    # early_exaggeration_iter: int
    # The number of iterations to run in the *early exaggeration* phase.
    early_exaggeration_iter: int = 250
    # early_exaggeration: float
    # The exaggeration factor to use during the *early exaggeration* phase.
    # Typical values range from 12 to 32.
    early_exaggeration: float = 12
    # n_iter: int
    # The number of iterations to run in the normal optimization regime.
    n_iter: int = 500
    # exaggeration: float
    # The exaggeration factor to use during the normal optimization phase.
    # This can be used to form more densely packed clusters and is useful
    # for large data sets.
    exaggeration: Optional[float] = None
    # dof: float
    # Degrees of freedom as described in Kobak et al. "Heavy-tailed kernels
    # reveal a finer cluster structure in t-SNE visualisations", 2019.
    dof: float = 1
    # theta: float
    # Only used when ``negative_gradient_method="bh"`` or its other aliases.
    # This is the trade-off parameter between speed and accuracy of the tree
    # approximation method. Typical values range from 0.2 to 0.8. The value 0
    # indicates that no approximation is to be made and produces exact results
    # also producing longer runtime.
    theta: float = 0.5
    # n_interpolation_points: int
    # Only used when ``negative_gradient_method="fft"`` or its other aliases.
    # The number of interpolation points to use within each grid cell for
    # interpolation based t-SNE. It is highly recommended leaving this value
    # at the default 3.
    n_interpolation_points: int = 3
    # min_num_intervals: int
    # Only used when ``negative_gradient_method="fft"`` or its other aliases.
    # The minimum number of grid cells to use, regardless of the
    # ``ints_in_interval`` parameter. Higher values provide more accurate
    # gradient estimations.
    min_num_intervals: int = 50
    # ints_in_interval: float
    # Only used when ``negative_gradient_method="fft"`` or its other aliases.
    # Indicates how large a grid cell should be e.g. a value of 3 indicates a
    # grid side length of 3. Lower values provide more accurate gradient
    # estimations.
    ints_in_interval: float = 1
    # initialization: Union[np.ndarray, str]
    # The initial point positions to be used in the embedding space. Can be a
    # precomputed numpy array, ``pca``, ``spectral`` or ``random``. Please
    # note that when passing in a precomputed positions, it is highly
    # recommended that the point positions have small variance
    # (std(Y) < 0.0001), otherwise you may get poor embeddings.
    initialization: Union[np.ndarray, Literal['pca', 'spectral', 'random']] = "pca"
    # metric: Union[str, Callable]
    # The metric to be used to compute affinities between points in the
    # original space.
    # "cosine", "euclidean",  "manhattan", "hamming", "dot", "l1", "l2", "taxicab"
    metric: Union[Literal["cosine", "euclidean", "manhattan", "hamming", "dot", "l1", "l2", "taxicab"],
                  Callable] = "cosine"
    # metric_params: dict
    # Additional keyword arguments for the metric function.
    metric_params: Optional[dict] = None
    # initial_momentum: float
    # The momentum to use during the *early exaggeration* phase.
    initial_momentum: float = 0.5
    # final_momentum: float
    # The momentum to use during the normal optimization phase.
    final_momentum: float = 0.8
    # max_grad_norm: float
    # Maximum gradient norm. If the norm exceeds this value, it will be
    # clipped. This is most beneficial when adding points into an existing
    # embedding and the new points overlap with the reference points,
    # leading to large gradients. This can make points "shoot off" from
    # the embedding, causing the interpolation method to compute a very
    # large grid, and leads to worse results.
    max_grad_norm: Optional[float] = None
    # max_step_norm: float
    # Maximum update norm. If the norm exceeds this value, it will be
    # clipped. This prevents points from "shooting off" from
    # the embedding.
    max_step_norm: float = 5
    # n_jobs: int
    # The number of threads to use while running t-SNE. This follows the
    # scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
    # all but one, etc.
    n_jobs: int = 1
    # affinities: openTSNE.affinity.Affinities
    # A precomputed affinity object. If specified, other affinity-related
    # parameters are ignored e.g. `perplexity` and anything nearest-neighbor
    # search related.
    affinities: Affinities = None
    # neighbors: str
    # Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
    # ``pynndescent``, ``approx``, or ``auto`` (default). ``approx`` uses Annoy
    # if the input data matrix is not a sparse object and if Annoy supports
    # the given metric. Otherwise it uses Pynndescent. ``auto`` uses exact
    # nearest neighbors for N<1000 and the same heuristic as ``approx`` for N>=1000.
    neighbors: str = "auto"
    # negative_gradient_method: str
    # Specifies the negative gradient approximation method to use. For smaller
    # data sets, the Barnes-Hut approximation is appropriate and can be set
    # using one of the following aliases: ``bh``, ``BH`` or ``barnes-hut``.
    # For larger data sets, the FFT accelerated interpolation method is more
    # appropriate and can be set using one of the following aliases: ``fft``,
    # ``FFT`` or ``ìnterpolation``.
    negative_gradient_method: str = "fft"
    # callbacks: Union[Callable, List[Callable]]
    # Callbacks, which will be run every ``callbacks_every_iters`` iterations.
    callbacks: Union[Callable, List[Callable]] = None
    # callbacks_every_iters: int
    # How many iterations should pass between each time the callbacks are
    # invoked.
    callbacks_every_iters: int = 50
    # random_state: Union[int, RandomState]
    # If the value is an int, random_state is the seed used by the random
    # number generator. If the value is a RandomState instance, then it will
    # be used as the random number generator. If the value is None, the random
    # number generator is the RandomState instance used by `np.random`.
    random_state: Union[int, RandomState] = None
    # verbose: bool
    verbose: bool = True


@dataclass
class AffinitiesParams:
    data: np.ndarray = None
    # method: str
    # Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
    # ``pynndescent``, ``approx``, or ``auto`` (default). ``approx`` uses Annoy
    # if the input data matrix is not a sparse object and if Annoy supports
    # the given metric. Otherwise it uses Pynndescent. ``auto`` uses exact
    # nearest neighbors for N<1000 and the same heuristic as ``approx`` for N>=1000.
    method: Literal['exact', 'annoy', 'hnswlib', 'pynndescent', 'approx', 'auto'] = "auto"
    # metric: Union[str, Callable]
    # The metric to be used to compute affinities between points in the
    # original space.
    metric: Union[Literal["cosine", "euclidean", "manhattan", "hamming", "dot", "l1", "l2", "taxicab"],
                  Callable] = "cosine"
    # metric_params: dict
    # Additional keyword arguments for the metric function.
    metric_params: Optional[dict] = None
    # symmetrize: bool
    # Symmetrize affinity matrix. Standard t-SNE symmetrizes the interactions
    # but when embedding new data, symmetrization is not performed.
    symmetrize: bool = True
    # n_jobs: int
    # The number of threads to use while running t-SNE. This follows the
    # scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
    # all but one, etc.
    n_jobs: int = -2
    # random_state: Union[int, RandomState]
    # If the value is an int, random_state is the seed used by the random
    # number generator. If the value is a RandomState instance, then it will
    # be used as the random number generator. If the value is None, the random
    # number generator is the RandomState instance used by `np.random`.
    random_state: Union[int, RandomState] = None
    # verbose: bool
    verbose: bool = False


@dataclass
class PerplexityParams(AffinitiesParams):
    # perplexity: float
    # Perplexity can be thought of as the continuous :math:`k` number of
    # nearest neighbors, for which t-SNE will attempt to preserve distances.
    perplexity: float = 30


@dataclass
class SigmaParams(AffinitiesParams):
    # sigma: float
    # The bandwidth to use for the Gaussian kernels in the ambient space.
    sigma: float = 1
    # k: int
    # The number of nearest neighbors to consider for each kernel.
    k: int = 30


@dataclass
class UniformParams(AffinitiesParams):
    # k_neighbors: int
    k_neighbors: int = 30


class OpenTSNEModel(Model):
    @classmethod
    def fit_data(cls, data=None, params: OpenTSNEParams = None):
        if params is None:
            params = OpenTSNEParams()
        if data is not None:
            data = np.array(data, dtype=np.float)
        else:
            raise AssertionError('Needs either data or input_file parameter!')

        model = TSNE(**asdict(params))
        return model.fit(data)

    @classmethod
    def _strategy_affinity(cls, data: Iterable, params: OpenTSNEParams, affinity: Type[Affinities],
                           affinity_params: Union[PerplexityParams, SigmaParams, UniformParams]):
        embedding: TSNEEmbedding
        affinities: Affinities

        prev_data = None

        for interval, (interval_data, interval_labels) in enumerate(data):
            logger.info(f'> Running OpenTSNE ({affinity.__name__}) for interval {interval} '
                        f'with {len(interval_data)} data points.')
            if interval == 0:
                affinity_params.data = interval_data
                affinities = affinity(**asdict(affinity_params))
                if params.initialization == "pca":
                    init = initialization.pca(
                        interval_data,
                        params.n_components,
                        random_state=params.random_state,
                        verbose=params.verbose,
                    )
                elif params.initialization == 'spectral':
                    init = initialization.spectral(
                        affinities.P,
                        params.n_components,
                        random_state=params.random_state,
                        verbose=params.verbose,
                    )
                else:
                    raise NotImplementedError('Init strategy no implemented.')

                embedding = TSNEEmbedding(
                    init, affinities,
                    negative_gradient_method='fft',
                    n_jobs=params.n_jobs
                )
                embedding.optimize(n_iter=params.early_exaggeration_iter, exaggeration=params.early_exaggeration,
                                   momentum=params.initial_momentum, inplace=True)
                embedding.optimize(n_iter=params.n_iter, exaggeration=params.exaggeration,
                                   momentum=params.final_momentum, inplace=True)
            else:
                P, neighbours, distances = affinities.to_new(interval_data[len(prev_data):], return_distances=True)
                init = initialization.weighted_mean(
                    interval_data[len(prev_data):], embedding, neighbours, distances
                )
                init = initialization.rescale(np.vstack((embedding, init)))
                affinity_params.data = interval_data
                affinities = affinity(**asdict(affinity_params))

                embedding = TSNEEmbedding(
                    init, affinities,
                    learning_rate=params.learning_rate,
                    negative_gradient_method='fft',
                    n_jobs=params.n_jobs
                )
                embedding.optimize(n_iter=params.n_iter, exaggeration=params.exaggeration,
                                   momentum=params.final_momentum, inplace=True)

            prev_data = interval_data
            yield interval_labels, np.array(embedding)

    @classmethod
    def strategy_perplexity(cls, data, params: OpenTSNEParams, strategy_params: PerplexityParams):
        yield from cls._strategy_affinity(data, params, PerplexityBasedNN, strategy_params)

    @classmethod
    def strategy_sigma(cls, data, params: OpenTSNEParams, strategy_params: SigmaParams):
        yield from cls._strategy_affinity(data, params, FixedSigmaNN, strategy_params)

    @classmethod
    def strategy_uniform(cls, data, params: OpenTSNEParams, strategy_params: UniformParams):
        yield from cls._strategy_affinity(data, params, Uniform, strategy_params)
