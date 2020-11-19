# -*- coding: utf-8 -*-

from .basic import *

"""
这三个类继承了 ReliefF 抽象类，主要的不同点是寻找最近邻样本集合的方式不同
分别实现抽象类中的抽象方法
但距离度量，特征评价的方式都一样
这个地方需要重点设计，什么的方法是正常方法，什么的方法是抽象方法，静态方法

ReliefFSupervised:
有监督下的 ReliefF 的算法
因样本空间中含有所有的样本标签，它直接在有标签的同类或者异类中寻找最近邻样本

ReliefFUnsupervised:
无监督下的 ReliefF 的算法
因样本空间中仅含有部分的样本标签，它在无标签样本集中寻找最近邻样本

ReliefFUnsupervisedImprove:
改进后无监督下的 ReliefF 的算法
因样本空间中仅含有部分的样本标签，它在无标签样本集加上与当前样本同类的有标签样本集合中寻找最近邻样本

"""


class ReliefFSupervised(ReliefF):
    """
    有监督下的 ReliefF 的算法
    样本空间中的所有样本都有标签
    当前被选中的样本，会在与其同标签的样本集中寻找最近邻的 k 个样本
    之后在与其不同标签的样本集中都去寻找最近邻的 k 个样本

    """

    def __init__(self, data, sample_rate):
        """
        :param data:
        数据集 pandas.DataFrame 或 pandas.core.frame.DataFrame 类型
        :param sample_rate:
        抽样比例，以一定比例从数据集中抽取样本
        """
        ReliefF.__init__(self, data, sample_rate)

    def fun(self):
        pass


class ReliefFUnsupervised(ReliefF):
    """
    无监督下的 ReliefF 的算法
    样本空间中的样本只有部分含有标签
    当前被选中的样本，会选择一个与当前样本同标签的最近邻样本，然后会在所有的无标签样本集中寻找最近邻的 k 个样本
    之后在与其不同标签的样本集中都去寻找一个最近邻样本，这些所有不同类的最近邻样本都会在所有的无标签样本集中寻找最近邻的 k 个样本

    """

    def __init__(self, data, sample_rate):
        """
        :param data:
        数据集 pandas.DataFrame 或 pandas.core.frame.DataFrame 类型
        :param sample_rate:
        抽样比例，以一定比例从数据集中抽取样本
        """
        ReliefF.__init__(self, data, sample_rate)

    def fun(self):
        pass


class ReliefFUnsupervisedImprove(ReliefF):
    """
    改进无监督下的 ReliefF 的算法
    样本空间中的样本只有部分含有标签
    当前被选中的样本，会选择一个与当前样本同标签的最近邻样本，然后会在
    所有的无标签样本集和与当前样本同类的所有有相同标签的样本集合(不含当前有标签的样本)的并集
    中寻找最近邻的 k 个样本
    之后在与其不同标签的样本集中都去寻找一个最近邻样本，这些所有不同类的最近邻样本都会在
    所有的无标签样本集和与当前最近邻样本的所有有相同标签的样本集(不含当前有标签的样本)的并集
    中寻找最近邻的 k 个样本

    """

    def __init__(self, data, sample_rate):
        """
        :param data:
        数据集 pandas.DataFrame 或 pandas.core.frame.DataFrame 类型
        :param sample_rate:
        抽样比例，以一定比例从数据集中抽取样本
        """
        ReliefF.__init__(self, data, sample_rate)

    def fun(self):
        pass
