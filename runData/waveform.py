# -*- coding: utf-8 -*-


import pandas
import operator
from typing import List
from core.dealData import *
from core.reliefF import *
from core.basic import *
from core.divide import *

original_path = "../data/originalDataSet/waveform/waveform.txt"
separator = " "
attribute_name = ['Attr1', 'Attr2', 'Attr3', 'Attr4', 'Attr5', 'Attr6', 'Attr7', 'Attr8', 'Attr9', 'Attr10', 'Attr11',
                  'Attr12', 'Attr13', 'Attr14', 'Attr15', 'Attr16', 'Attr17', 'Attr18', 'Attr19', 'Attr20', 'Attr21',
                  "Label"]
''''''

"""
以 csv_path 路径的csv文件，以 attribute_dict 为属性字典，以 remove_rate 去除比例
使用 DealData 类，对连续特征归一化，离散特征概率化后归一化得到处理好后的两份数据集
有监督数据集路径为 data_path_supervised
半监督数据集路径为 data_path_unSupervised
"""

csv_path = "../data/originalDataSet/waveform/waveform.csv"
attribute_dict = {'Attr1': 0, 'Attr2': 0, 'Attr3': 0, 'Attr4': 0, 'Attr5': 0, 'Attr6': 0, 'Attr7': 0, 'Attr8': 0,
                  'Attr9': 0, 'Attr10': 0, 'Attr11': 0, 'Attr12': 0, 'Attr13': 0, 'Attr14': 0, 'Attr15': 0, 'Attr16': 0,
                  'Attr17': 0, 'Attr18': 0, 'Attr19': 0, 'Attr20': 0, 'Attr21': 0, "Label": -1
                  }
remove_rate = 0.9

data_path_supervised = "../data/processedDataSet/waveform/waveform_supervised.csv"
data_path_unSupervised = "../data/processedDataSet/waveform/waveform_unSupervised.csv"


def fun2():
    """
    使用 DealData 类
    :return:
    """
    d = DealData(csv_path, attribute_dict, remove_rate)
    '''使用 variance_data 归一化'''
    d.variance_data()
    '''使用 deal_data() 以 remove 去除标签得到半监督数据集'''
    d.deal_data()


''''''

"""
以 divide_rate 划分数据集
调用 DivideData 类
"""

divide_rate = 1


def fun3():
    """
    以 divide_rate 划分数据集，这里不调用这个方法，但保留
    :return:
    """


''''''

"""
KMeans 算法研究：
应用在半监督数据集 data_path_unSupervised 上
三种 KMeans 算法，得到三次 KMeans 的结果，作图，比较结果
使用改进 KMeans 算法得到的聚类结果，在半监督数据集聚类中心以 sample 抽取数据
具体的做法是得到需要抽取的数据的索引
将该索引应用到有监督 data_path_supervised 和半监督数据集 data_path_unSupervise 上
得到的新的文件，其路径是位于原本路径的子文件夹 kMeans 下，文件名保持相同
"""

sample = 0.2
final_path_supervised = "../data/processedDataSet/waveform/kMeans/waveform_supervised.csv"
final_path_unSupervised = "../data/processedDataSet/waveform/kMeans/waveform_unSupervised.csv"


def gain_label_index() -> dict:
    """
    gain label_index
    :return:
    label_index:
    """
    label_index = dict()
    data = pandas.read_csv(data_path_supervised)
    g = data.groupby(attribute_name[-1]).groups
    for label, value in g.items():
        label_index[label] = list(value)
    return label_index


def fun_kMeans():
    """
    调用 MKMeans 算法
    :return:
    """
    print("------K_Means------")
    km = MKMeans(pandas.read_csv(data_path_unSupervised), sample)
    ''''''
    label_index = gain_label_index()
    km.k_means()
    print(km.centroids)
    print(km.cluster_class)
    km.print_result(label_index)
    km.k_means_pluses()
    print(km.centroids)
    print(km.cluster_class)
    km.print_result(label_index)
    km.k_means_m()
    print(km.centroids)
    print(km.cluster_class)
    km.print_result(label_index)
    ''''''
    km.data_divide()
    '''得到最终的文件路径'''
    final_path_supervised = deal_path(data_path_supervised)
    final_path_unSupervised = deal_path(data_path_unSupervised)
    index_list = list()
    for value in km.final_data.values():
        index_list.extend(value)
    data_supervised = pandas.read_csv(data_path_supervised)
    data_supervised.iloc[index_list, :].to_csv(final_path_supervised, index=False, sep=",")
    data_unSupervised = pandas.read_csv(data_path_unSupervised)
    data_unSupervised.iloc[index_list, :].to_csv(final_path_unSupervised, index=False, sep=",")


''''''

"""
Relief 算法
使用
ReliefSupervised
ReliefFUnsupervised
ReliefFUnsupervisedImprove
类得到每个特征的权重，排序，选取一定的特征子集，得到三个特征子集
用该特征子集，在 data_path_supervised 上进行十折交叉验证，得到三个结果
"""

relief_rate = 0.3
k = 3


def relief_f():
    """
    调用
    ReliefSupervised
    ReliefFUnsupervised
    ReliefFUnsupervisedImprove
    :return:
    """
    print("------ReliefF------")
    r_supervised = ReliefFSupervised(pandas.read_csv(final_path_supervised),
                                     attribute_dict, relief_rate, k)
    r_unSupervised = ReliefFUnsupervised(pandas.read_csv(final_path_unSupervised),
                                         attribute_dict, relief_rate, k)
    r_unSupervised_improve = ReliefFUnsupervisedImprove(pandas.read_csv(final_path_unSupervised),
                                                        attribute_dict, relief_rate, k)
    res_supervised = dict(r_supervised.relief_f())
    res_supervised = sorted(res_supervised.items(), key=lambda item: item[1], reverse=True)
    print(res_supervised)
    res_unsupervised = dict(r_unSupervised.relief_f())
    res_unsupervised = sorted(res_unsupervised.items(), key=lambda item: item[1], reverse=True)
    print(res_unsupervised)
    res_unsupervised_improve = dict(r_unSupervised_improve.relief_f())
    res_unsupervised_improve = sorted(res_unsupervised_improve.items(), key=lambda item: item[1], reverse=True)
    print(res_unsupervised_improve)


if __name__ == "__main__":
    ''''''
    # fun1()
    # print(csv_path)
    fun2()
    fun_kMeans()
    relief_f()
