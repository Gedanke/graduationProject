# -*- coding: utf-8 -*-

import numpy
import pandas
from typing import List, Dict

"""
该部分的任务是将大的数据集划分为若干块，从每一块中按一定比例抽取样本，汇总得到一个新的样本集合
读取的 pandas.DataFrame 类型数据，其中数据集已经经过清洗，离散特征熵值连续化，所有特征都已经归一化
处理后，数据集会被划分到不同块中
会从每块中按一定的比例抽取数据集

KDTree：
使用 Kd-Tree 算法将数据划分到叶子节点中
https://blog.csdn.net/qq_33690156/article/details/52452950

KMeans:
使用 K-Means 算法将数据集按标签类别数进行划分
https://www.cnblogs.com/pinard/p/6164214.html

"""


class KDTree(object):
    """
    Kd-Tree 算法
    首先在方差最大的特征维度上划分数据集，之后在剩下的特征中选择方差大的，以此类推
    最后所有的样本都被划分到叶子节点中，当然一个叶子节点会含有多个样本
    在每个叶子节点中按比例抽取样本

    """

    def __init__(self, data: pandas.DataFrame):
        """
        :param data:
        数据集 pandas.DataFrame 或 pandas.core.frame.DataFrame 类型
        """
        self.data = data


class KMeans(object):
    """
    在选取原始的 k 个基本点时，先将数据集中含有标签的样本按类别划分成不同集合
    在每一个类别集合中，选取该类所有样本的中心点为该类别的初始点
    最后每个类别都有一个初始基本点，之后进行聚类，k 类
    在每类中按比例抽取样本

    """

    def __init__(self, data: pandas.DataFrame, parameter_list: Dict[str, str]):
        """
        :param data:
        数据集 pandas.DataFrame 或 pandas.core.frame.DataFrame 类型
        :param parameter_list:
        其他的参数字典
        """
        self.data = data
        self.parameter_list = parameter_list
        self.center_point = list()
        self.k = 0
