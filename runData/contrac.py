# -*- coding: utf-8 -*-


import pandas
import operator
from typing import List
from core.dealData import *
from core.reliefF import *
from core.basic import *
from core.divide import *

original_path = "../originalDataSet/contrac/contrac.txt"
separator = ","
attribute_name = [
    "wife_age", "wife_education", "husband_education", "number", "wife_religion", "wife_working", "husband_occupation",
    "standard", "media_exposure", "label"
]

''''''

"""
以 csv_path 路径的csv文件，以 attribute_dict 为属性字典，以 remove_rate 去除比例
使用 DealData 类，对连续特征归一化，离散特征概率化后归一化得到处理好后的两份数据集
有监督数据集路径为 data_path_supervised
半监督数据集路径为 data_path_unSupervised
"""

csv_path = "../originalDataSet/contrac/contrac.csv"
attribute_dict = {
    "wife_age": 0, "wife_education": 1, "husband_education": 1, "number": 0, "wife_religion": 1, "wife_working": 1,
    "husband_occupation": 1, "standard": 1, "media_exposure": 1, "label": -1
}
remove_rate = 0.8

data_path_supervised = ""
data_path_unSupervised = ""


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

sample = 0.3
final_path_supervised = ""
final_path_unSupervised = ""


def fun_kMeans():
    """
    调用 MKMeans 算法
    :return:
    """
    print("------K_Means------")
    km = MKMeans(pandas.read_csv(data_path_unSupervised), sample)
    ''''''
    km.k_means()
    print(km.centroids)
    print(km.cluster_class)
    km.k_means_pluses()
    print(km.centroids)
    print(km.cluster_class)
    km.k_means_m()
    print(km.centroids)
    print(km.cluster_class)
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
    res_supervised = r_supervised.relief_f()
    print(res_supervised)
    res_unsupervised = r_unSupervised.relief_f()
    print(res_unsupervised)
    res_unsupervised_improve = r_unSupervised_improve.relief_f()
    print(res_unsupervised_improve)


if __name__ == "__main__":
    # fun1()
    # print(csv_path)
    fun2()
