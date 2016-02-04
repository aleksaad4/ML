import math

import numpy
from scipy.spatial.distance import euclidean


def loss(X, y, w, C):
    l = len(y)
    value = 0.0
    for (x, y_i) in zip(X, y):
        value += math.log(1.0 + math.exp(-y_i * numpy.dot(x, w)))
    value /= l
    value += C / 2.0 * numpy.sum(numpy.square(w))
    return value


def update(X, y, w, k, C):
    l = len(y)
    value = numpy.zeros(len(w))
    for idx in range(0, l):
        x = X.iloc[idx]
        y_i = y[idx]
        for i, w_i in enumerate(w):
            value[i] += y_i * x.loc[i + 1] * (1 - 1.0 / (1 + math.exp(-y_i * numpy.dot(x, w))))

    value += w - k * C * w
    return value


def gd(X, y, k=0.1, init_w=None, C=0, eps=1e-5, max_iter=10000):
    if init_w is None:
        w = numpy.zeros(len(X.iloc[1]))
    else:
        w = init_w.copy()

    diff = None
    while diff is None or diff > eps:
        cur_w = update(X, y, w, k, C)
        diff = euclidean(w, cur_w)
        print diff
        w = cur_w

    return w
