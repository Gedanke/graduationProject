# -*- coding: utf-8 -*-

from mnist.loader import MNIST
from sklearn.cluster import KMeans
import pandas
import numpy

# data = MNIST('datasets')
# X_train, y_train = data.load_training()
#
# # do the clustering
# k_means = KMeans(n_clusters=len(numpy.unique(y_train)))
# k_means.fit(X_train)
# labels = k_means.labels_
#
# predict = k_means.predict(data)
# data['cluster'] = predict
# pandas.tools.plotting.parallel_coordinates(data, 'cluster')
from functools import cmp_to_key

d = [(0, 6, {8, 10, 11, 13, 15, 16}), (1, 11, {0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 14})]

print(d)


def func(x, y):
    if x[1] < y[1]:
        return -1
    elif x[1] == y[1]:
        return 0
    else:
        return 1


sorted(d, key=lambda x: x[1])
l = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 14]
print(len(d[1][2]))
