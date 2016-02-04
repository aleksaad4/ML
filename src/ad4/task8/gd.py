import math

import numpy
from scipy.spatial.distance import euclidean


def proba(X, w):
    l = len(X)
    p = []
    for idx in range(0, l):
        x = X.iloc[idx]
        p.append(1.0 / (1 + math.exp(-1 * numpy.dot(x, w))))
    return p


def loss(X, y, w, C):
    l = len(y)
    value = 0.0
    for idx in range(0, l):
        x = X.iloc[idx]
        y_i = y[idx]
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
    value *= k / l
    value += w - k * C * w
    return value


def gd(X, y, k=0.1, init_w=None, C=0, eps=1e-5, max_iter=10000):
    if init_w is None:
        w = numpy.zeros(len(X.iloc[1]))
    else:
        w = init_w.copy()

    diff = None
    it = 0
    while diff is None or diff > eps or it > max_iter:
        cur_w = update(X, y, w, k, C)
        diff = euclidean(w, cur_w)
        it += 1
        w = cur_w

    print "Iter count [" + str(it) + "]"
    return w
