# -*- coding: utf-8 -*-

from .basic import *

"""
ReliefFSupervised:
有监督下的 ReliefF 的算法
因样本空间中含有所有的样本标签，它直接在同类或者异类中寻找最近邻样本

ReliefFUnsupervised:
无监督下的 ReliefF 的算法
因样本空间中仅含有部分的样本标签，它在无标签样本集中寻找最近邻样本

ReliefFUnsupervisedImprove:
改进后无监督下的 ReliefF 的算法
因样本空间中仅含有部分的样本标签，它在无标签样本集加上同类有标签样本的集合中寻找最近邻样本
"""


class ReliefFSupervised(ReliefF):
    def __init__(self, data, sample_rate):
        """
        :param data: 数据集 pandas.DataFrame 或 pandas.core.frame.DataFrame 类型
        :param sample_rate: 抽样比例，以一定比例从数据集中抽取样本
        """
        ReliefF.__init__(self, data, sample_rate)


class ReliefFUnsupervised(ReliefF):
    def __init__(self, data, sample_rate):
        """
        :param data: 数据集 pandas.DataFrame 或 pandas.core.frame.DataFrame 类型
        :param sample_rate: 抽样比例，以一定比例从数据集中抽取样本
        """
        ReliefF.__init__(self, data, sample_rate)


class ReliefFUnsupervisedImprove(ReliefF):
    def __init__(self, data, sample_rate):
        """
        :param data: 数据集 pandas.DataFrame 或 pandas.core.frame.DataFrame 类型
        :param sample_rate: 抽样比例，以一定比例从数据集中抽取样本
        """
        ReliefF.__init__(self, data, sample_rate)
