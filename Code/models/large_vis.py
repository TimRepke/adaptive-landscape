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


class LargeVisModel(Model):
    @staticmethod
    def write_temp_file(file_name, data):
        with open(file_name, 'w') as f:
            f.write(f'{len(data)}\t{len(data[0])}\n')
            [f.write('\t'.join([f'{di:.5f}' for di in d]) + '\n') for d in data]

    @staticmethod
    def fit_interval(data=None, input_file=None, n_threads=-1, n_propagations=-1, alpha=-1, gamma=-1, perplexity=-1,
                     n_trees=-1, n_negatives=-1, n_neighbours=-1):
        if data is not None:
            data = np.array(data, dtype=np.float)
            LargeVis.loadarray(np.array(data))
        elif input_file is not None:
            LargeVis.loadfile(input_file)
        else:
            raise AssertionError('Needs either data or input_file parameter!')

        Y = LargeVis.run(2,  # output_dimension
                         n_threads,  # threads_number
                         -1,  # training_samples
                         n_propagations,  # propagations_number
                         alpha,  # learning_rate
                         n_trees,  # rp_trees_number
                         n_negatives,  # negative_samples_number
                         n_neighbours,  # neighbors_number
                         gamma,  # gamma
                         perplexity  # perplexity
                         )
        return Y
