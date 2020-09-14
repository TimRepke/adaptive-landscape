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
from models import Model


class LargeVisModel(Model):
    @staticmethod
    def fit_interval(data=None, input_file=None, threads=-1, n_propagations=-1, alpha=-1, gamma=-1, perplexity=-1,
                     n_trees=-1, n_negatives=-1, n_neighbours=-1):
        if data is not None:
            LargeVis.loaddata(data)
        elif input_file is not None:
            pass
        else:
            raise AssertionError('Needs either data or input_file parameter!')

        Y = LargeVis.run(output_dimension=2, threads_number=threads, training_samples=-1,
                         propagations_number=n_propagations, learning_rate=alpha, rp_trees_number=n_trees,
                         negative_samples_number=n_negatives, neighbors_number=n_neighbours,
                         gamma=gamma, perplexity=perplexity)
        return Y
