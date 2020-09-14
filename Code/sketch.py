from numba import njit
from numba.typed import List
from data_readers.mnist import MNIST
import time


def do(labels, n_labels):
    ret = [[]] * n_labels
    for i, label in enumerate(labels):
        ret[label].append(i)
    return ret


def do2(labels, n_labels):
    ret = [[]] * n_labels
    [ret[label].append(i) for i, label in enumerate(labels)]

    return ret


@njit
def jdo(labels, n_labels):
    ret = List()
    for _ in range(n_labels):
        ret.append(List('int'))
    for i, label in enumerate(labels):
        ret[label].append(i)
    return ret


def dumb(labels, n_labels):
    ret = [[i for i, ll in enumerate(labels) if ll == lab] for lab in range(n_labels)]


m = MNIST('../RawData/MNIST')

l = m.get_labels().astype(int).tolist()

start_time = time.time()
do(l, len(m.data_labels()))
used_time = time.time() - start_time
print(f'split each line at tab took {used_time:.3f}s')
start_time = time.time()
do(l, len(m.data_labels()))
used_time = time.time() - start_time
print(f'split each line at tab took {used_time:.3f}s')
start_time = time.time()
dumb(l, len(m.data_labels()))
used_time = time.time() - start_time
print(f'split each line at tab took {used_time:.3f}s')

jdumb = njit()(dumb)
jl = m.get_labels().astype(int)  # List(l)
start_time = time.time()
jdumb(jl, len(m.data_labels()))
used_time = time.time() - start_time
print(f'split each line at tab took {used_time:.3f}s')
start_time = time.time()
jdumb(jl, len(m.data_labels()))
used_time = time.time() - start_time
print(f'split each line at tab took {used_time:.3f}s')

start_time = time.time()
jdo(jl, len(m.data_labels()))
used_time = time.time() - start_time
print(f'split each line at tab took {used_time:.3f}s')

start_time = time.time()
jdo(jl, len(m.data_labels()))
used_time = time.time() - start_time
print(f'split each line at tab took {used_time:.3f}s')
