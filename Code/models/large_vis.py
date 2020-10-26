# import LargeVis
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-fea', default = 1, type = int,
#                     help='whether to visualize high-dimensional feature vectors or networks')
# parser.add_argument('-input', default = '', help = 'input file')
# parser.add_argument('-output', default = '', help = 'output file')
# parser.add_argument('-outdim', default = -1, type = int, help = 'output dimensionality')
# parser.add_argument('-threads', default = -1, type = int, help = 'number of training threads')
# parser.add_argument('-samples', default = -1, type = int, help = 'number of training mini-batches')
# parser.add_argument('-prop', default = -1, type = int, help = 'number of propagations')
# parser.add_argument('-alpha', default = -1, type = float, help = 'learning rate')
# parser.add_argument('-trees', default = -1, type = int, help = 'number of rp-trees')
# parser.add_argument('-neg', default = -1, type = int, help = 'number of negative samples')
# parser.add_argument('-neigh', default = -1, type = int, help = 'number of neighbors in the NN-graph')
# parser.add_argument('-gamma', default = -1, type = float, help = 'weight assigned to negative edges')
# parser.add_argument('-perp', default = -1, type = float, help = 'perplexity for the NN-grapn')
#
# args = parser.parse_args()
#
# if args.fea == 1:
#     LargeVis.loadfile(args.input)
# else:
#     LargeVis.loadgraph(args.input)
#
# Y = LargeVis.run(args.outdim, args.threads, args.samples, args.prop, args.alpha, args.trees,
#                  args.neg, args.neigh, args.gamma, args.perp)
#
# LargeVis.save(args.output)
import LargeVis
import numpy as np
from models import Model
from dataclasses import dataclass, asdict
import tempfile
import logging
import os

logger = logging.getLogger('LargeVis')


@dataclass
class LargeVisParams:
    output_dimension: int = 2
    threads_number: int = -1
    training_samples: int = -1
    propagations_number: int = -1
    learning_rate: float = -1
    rp_trees_number: int = -1
    negative_samples_number: int = -1
    neighbors_number: int = -1
    gamma: float = -1
    perplexity: int = -1


class LargeVisModel(Model):

    @staticmethod
    def _cleanup():
        os.remove('annoy_index_file')

    @staticmethod
    def _write_temp_file(data):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(f'{len(data)}\t{len(data[0])}\n')
            [f.write('\t'.join([f'{di:.5f}' for di in d]) + '\n') for d in data]
            return f.name

    @classmethod
    def fit_data(cls, data=None, params: LargeVisParams = None):
        logger.info('Writing data temporarily to file...')
        file_name = cls._write_temp_file(data)
        LargeVis.loadfile(file_name)
        logger.debug(f'Stored to {file_name}')

        logger.info('Running LargeVis...')
        points = LargeVis.run(*asdict(params).values())

        logger.debug('Removing temporary files again...')
        os.remove(file_name)
        cls._cleanup()

        return points
