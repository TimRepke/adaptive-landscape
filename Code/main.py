from data_readers.mnist import MNIST
from data_readers.fashion_mnist import FashionMNIST
import gzip
from matplotlib import pyplot as plt
import numpy as np
import umap
import umap.plot

m = MNIST('../RawData/MNIST')
fm = FashionMNIST('../RawData/FashionMNIST')

digits = [i[0].reshape((-1,)) for i in m.get_data()][:1000]
labels = np.array([i[1] for i in m.get_data()][:1000])
print(digits[0])

fdigits = [i[0].reshape((-1,)) for i in fm.get_data()][:1000]
flabels = np.array([i[1] for i in fm.get_data()][:1000])

fm.show(fm.create_sprite_image())
fm.show(fm.get_data()[0][0])

# embedding = umap.UMAP(n_neighbors=5,
#                      min_dist=0.3,
#                      metric='correlation').fit_transform(m.get_data())

mapper = umap.UMAP(n_neighbors=5,
                   min_dist=0.1,
                   metric='euclidean').fit(digits)
umap.plot.points(mapper, labels=labels)
