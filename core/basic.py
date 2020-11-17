# -*- coding: utf-8 -*-

import pandas
from abc import ABCMeta, abstractmethod

"""
ReliefF是一个抽象类，它不可以被实例化
可以由 ReliefFSupervised ,ReliefFUnsupervised 和 ReliefFUnsupervisedImprove 类继承并实现其中的若干个抽象方法

ReliefF 类是 ReliefF 算法的核心类，其中已经实现了大多数方法
但由于有监督下的，无监督下的，改进无监督下的 ReliefF 算法在搜索最近邻样本时，或者其他部分略有不同
因此该部分定义为抽象方法，交给不同的子类去实现这部分的细节
而除此之外的其他流程基本上相同
"""


class ReliefF(metaclass=ABCMeta):
    """
    ReliefF 算法可以处理多分类，不完整数据集，算法(有监督)流程：
    每次随即选择一个样本，在与当前样本相同标签的样本集中选择 k 个最近邻样本
    在与当前样本每个不同标签的样本集中分别寻找 k 个最近邻样本
    """

    def __init__(self, data: pandas.DataFrame, sample_rate: float):
        """
        :param data: 数据集 pandas.DataFrame 或 pandas.core.frame.DataFrame 类型
        :param sample_rate: 抽样比例，以一定比例从数据集中抽取样本
        """
        self.data = data
        self.sample_rate = sample_rate

    @abstractmethod
    def fun(self):
        pass
