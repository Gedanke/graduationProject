# -*- coding: utf-8 -*-

import pandas
import operator
import numpy
from typing import List
from core.dealData import *
from core.reliefF import *
from core.basic import *
from core.divide import *

original_path = "../originalDataSet/testData/xigua.txt"
separator = ","
attribute_name = [
    "色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率", "好瓜"
]


def fun1():
    t = TransformData(original_path, separator, attribute_name)
    t.mine_deal()


csv_path = "../originalDataSet/testData/xigua.csv"
attribute_dict = {
    "色泽": 1, "根蒂": 1, "敲声": 1, "纹理": 1, "脐部": 1, "触感": 1, "密度": 0, "含糖率": 0,
    "好瓜": -1
}
remove_rate = 0.8


def fun2():
    d = DealData(csv_path, attribute_dict, remove_rate)
    d.variance_data()
    # print(d.attribute_variance)
    d.deal_data()


data_path = "../processedDataSet/testData/xigua_supervised.csv"
data_path_ = "../processedDataSet/testData/xigua_unSupervised.csv"
divide_rate = 0.5


def fun3():
    dd = DivideData(data_path, divide_rate)


sample_rate = 0.5

k = 3


def relief_f():
    r_supervised = ReliefFSupervised(pandas.read_csv(data_path), attribute_dict,
                                     sample_rate, k)
    r_unsupervised = ReliefFUnsupervised(pandas.read_csv(data_path_),
                                         attribute_dict, sample_rate, k)
    r_unsupervised_improve = ReliefFUnsupervisedImprove(pandas.read_csv(data_path_),
                                                        attribute_dict, sample_rate, k)

    res_supervised = r_supervised.relief_f()
    print(res_supervised)
    res_unsupervised = r_unsupervised.relief_f()
    print(res_unsupervised)
    res_unsupervised_improve = r_unsupervised_improve.relief_f()
    print(res_unsupervised_improve)


def test():
    data = pandas.read_csv(data_path_)
    key_ = "色泽"
    d = data.groupby(key_).groups
    d_dict = dict()
    min_value = 1.1
    max_value = -0.1
    for key, value in d.items():
        print(key, value)
        d_dict[key] = len(value) / 17
        # print(type(d_dict[key]))
        if d_dict[key] > max_value:
            max_value = d_dict[key]
        if d_dict[key] < min_value:
            min_value = d_dict[key]
    print(d_dict)
    print(min_value, max_value)
    avg = 1 / len(d_dict)
    s = 0
    for key, value in d_dict.items():
        s += pow(abs(value - avg), 2)
        d_dict[key] = (value - min_value) / (max_value - min_value)
    print(s)
    print(d_dict)
    data[key_].replace(d_dict, inplace=True)
    print(data)


def test1():
    data = pandas.read_csv(data_path_)
    # print(data)
    attribute_list = list(data.columns)
    '''['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '好瓜']'''
    attribute_list.remove(attribute_list[-1])
    l = ['密度', '含糖率']
    row = data.iloc[0]
    distance_dict = {
        index: abs(row[l] - data.iloc[index][l]).sum()
        for index in range(17)
    }
    sim_list = sorted(distance_dict.items(),
                      key=lambda item: item[1], reverse=False)

    print(distance_dict)

    print(max(distance_dict, key=distance_dict.get))


def test2():
    print("---numpy---")
    a = [
        [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]
    ]
    data_array = numpy.array(a)
    print(data_array.shape)
    print(type(data_array))


def test_Kmeans():
    print("---K_Means---")
    km = MKMeans(pandas.read_csv(data_path_))
    # print(km.k)
    # print(type(km.center_point))
    # print(type(km.array_data))
    # print(numpy.array(km.center_point))
    print(km.k_means())
    print(km.k_means_pluses())
    print(km.k_means_m())

    a = [
        [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]
    ]
    # print(type(numpy.mat(a)))
    centroids = numpy.mat(numpy.zeros((2, 8)))
    # print(centroids)


if __name__ == "__main__":
    # fun1()
    # fun2()
    # fun3()
    # relief_f()
    test_Kmeans()
    # test2()
