import numpy as np
from models import Model
import enum

import openTSNE
import logging

logger = logging.getLogger('generators')


class OpenTSNEModel(Model):

    @staticmethod
    def fit_data():
        """
        """
        if data is not None:
            data = np.array(data, dtype=np.float)
        elif input_file is not None:
            raise AttributeError('not implemented')
        else:
            raise AssertionError('Needs either data or input_file parameter!')

        Y = None  # TODO

        return Y

    def fit_intervals(self, strategy, data):
        pass
