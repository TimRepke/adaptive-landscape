from datasets.mnist import MNIST
from datasets.fashion_mnist import FashionMNIST

import numpy as np
from data_handlers.generators import temporal, SamplingReference, accumulate
from data_handlers.disk import store_result
from models.large_vis import LargeVisModel

m = MNIST('../RawData/MNIST')
print('data loaded')
# fm = FashionMNIST('../RawData/FashionMNIST')

temporal_data, temporal_labels = temporal(m,
                                          label_dist=[{3: 0.8}, {3: 0.8}, {3: 0.8, 1: 20}],
                                          target_size=5000,
                                          sampling_reference=SamplingReference.LABEL_COUNT)

for interval, (interval_data, interval_labels) in enumerate(accumulate(temporal_data, temporal_labels, generate=True)):
    interval_data = [i.tolist() for i in interval_data]

    y = LargeVisModel.fit_interval(data=interval_data)
    store_result(y, interval_labels, 'mnist', 'largevis', interval)

# digits = [i[0].reshape((-1,)) for i in m.get_data()][:1000]
# labels = np.array([i[1] for i in m.get_data()][:1000])
# print(digits[0])
#
# fdigits = [i[0].reshape((-1,)) for i in fm.get_data()][:1000]
# flabels = np.array([i[1] for i in fm.get_data()][:1000])
# print()
# fm.show(fm.create_sprite_image())
# fm.show(fm.get_data()[0][0])
#
# # embedding = umap.UMAP(n_neighbors=5,
# #                      min_dist=0.3,
# #                      metric='correlation').fit_transform(m.get_data())
#
# mapper = umap.UMAP(n_neighbors=5,
#                    min_dist=0.1,
#                    metric='euclidean').fit(digits)
# umap.plot.points(mapper, labels=labels)
