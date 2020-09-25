import numpy as np
from models import Model
from externals.bhtsne import bhtsne


class BHtSNEModel(Model):

    @staticmethod
    def fit_interval(data=None, input_file=None,
                     no_dims=2, perplexity=50, theta=0.5, randseed=-1,
                     verbose=False, initial_dims=50, use_pca=True, max_iter=1000):
        """Run tSNE based on the Barnes-HT algorithm

        Parameters:
        ----------
        data: file or numpy.array
            The data used to run TSNE, one sample per row
        no_dims: int
        perplexity: int
        randseed: int
        theta: float
        initial_dims: int
        verbose: boolean
        use_pca: boolean
        max_iter: int
        """
        if data is not None:
            data = np.array(data, dtype=np.float)
        elif input_file is not None:
            raise AttributeError('not implemented')
        else:
            raise AssertionError('Needs either data or input_file parameter!')

        Y = bhtsne.run_bh_tsne(data, no_dims=no_dims, perplexity=perplexity, theta=theta, randseed=randseed,
                               verbose=verbose, initial_dims=initial_dims, use_pca=use_pca, max_iter=max_iter)
        return Y
