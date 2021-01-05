# -*- coding: utf-8 -*-

import pandas
import random
from abc import ABCMeta, abstractmethod
from typing import List, Dict

"""
ReliefF 是一个抽象类，它不可以被实例化
可以由 ReliefFSupervised, ReliefFUnsupervised 和 ReliefFUnsupervisedImprove 类继承并实现其中的若干个抽象方法

ReliefF 类是 ReliefF 算法的核心类，其中已经实现了大多数方法
但由于有监督下的，无监督下的，改进无监督下的 ReliefF 算法在搜索最近邻样本时，或者其他部分略有不同
因此该部分定义为抽象方法，交给不同的子类去实现这部分的细节
而除此之外的其他流程基本上相同

"""


class ReliefF(metaclass=ABCMeta):
    """
    ReliefF 算法可以处理多分类，不完整，含有噪声的数据集，算法(有监督)流程：
    迭代过程中，每次随机选择一个样本，在与当前样本相同标签的样本集中选择 k 个最近邻样本
    在与当前样本每个不同标签的样本集中分别寻找 k 个最近邻样本
    计算每个特征在一次迭代过程中的权重得分
    连续特征权重得分按两个属性间的距离计算
    离散特征权重得分是非 0 即 1
    
    """

    def __init__(self, data: pandas.DataFrame, attribute_dict: Dict[str, int], sample_rate: float, k: int):
        """
        :param data:
        数据集 pandas.DataFrame 或 pandas.core.frame.DataFrame 类型
        :param attribute_dict:
        attribute_dict 是属性名字典
        :param sample_rate:
        抽样比例，以一定比例从数据集中抽取样本
        :param k:
        ReliefF 算法里面要寻找的近邻 k
        """
        self.data = data
        self.attribute_dict = attribute_dict
        self.sample_rate = sample_rate
        self.k = k
        '''属性列表，包含标签名'''
        self.attribute_list = list(self.data.columns)
        '''标签名'''
        self.label_name = self.attribute_list[-1]
        '''属性列表，去除了标签名'''
        self.attribute_list.remove(self.label_name)
        '''有监督标签集合，此时可能含有无标签标记 None，无监督算法可以在抽象方法 init_abstract_msg 里去除 None'''
        # self.label_set = set(self.data[self.label_name])
        self.label_set = set()
        '''样本数量'''
        self.sample_num = len(self.data)
        '''特征权重向量'''
        self.feature_vector = dict()
        '''抽样样本索引列表'''
        self.sample_index = list()
        '''*** abstractmethod ***'''
        '''近邻搜索空间，键为标签，值为样本索引'''
        self.search_space = dict()
        '''有监督算法里是每个标签含有样本的数量，用来计算加权过程的系数'''
        '''无监督算法因为含有 None，其是 self.search_space 去除了 None 键值对后的字典'''
        self.label_dict = dict()

    @abstractmethod
    def init_abstract_msg(self):
        """
        初始化相关参数，这些参数在抽象方法里，即需要不同子类去实现
        self.sample_index: 抽样样本索引列表
        self.label_set: 标签集合
        self.search_space: 近邻搜索空间，键为标签，值为样本索引
        self.label_nums: 每个标签的数量
        :return:
        """
        '''self.sample_index'''
        '''有监督的在全体样本上抽取，而无监督的只在有标签的样本上抽取'''
        '''这里写的是有监督的'''
        sample = int(round(self.sample_num * self.sample_rate))
        self.sample_index = random.sample(list(range(self.sample_num)), sample)
        '''self.label_set'''
        '''self.search_space'''
        '''https://www.yiibai.com/pandas/python_pandas_groupby.html'''
        '''self.label_dict'''
        '''将数据集按 self.label_name 分组，键为标签，值为相同标签样本的索引，已经是字典序了'''
        label_groups = self.data.groupby(self.label_name).groups
        for label, value in label_groups.items():
            '''收集标签'''
            self.label_set.add(label)
            '''收集标签下样本的索引'''
            self.search_space[label] = list(value)
            '''收集标签下的样本索引长度，即每个样本的数量'''
            self.label_dict[label] = len(value)

    @abstractmethod
    def gain_near_miss_param(self, label_r: str, label_c: str) -> float:
        """
        不同类待乘的系数，不同子类实现的方式不一样
        返回与 label_r 不同标签的 label_c 特征加权时的系数
        :param label_r: 当前样本的标签
        :param label_c: 与 label_r 不同标签样本的标签 label_c 
        :return:
        param: 待乘的系数
        """
        param = 1
        return param

    @abstractmethod
    def distance(self, row: pandas.Series, label: str) -> List[int]:
        """
        在当前样本 row 搜索空间里寻找最近邻居样本，
        不同子类去寻找最近邻居样本的方法略有不同，即需要不同子类去实现
        :param row: 当前样本
        :param label: 在该 label 下搜索 k 个最近邻样本
        :return:
        sim: k 个最近邻样本的索引
        """
        '''有监督，无监督，改进无监督搜索用于加权的 k 个最近邻样本略有不同'''
        sim = list()
        return sim

    def get_neighbor(self, row: pandas.Series) -> [List[int], Dict[str, List[int]]]:
        """
        返回用于加权的同类 k 个最近邻样本，以及每个不同类的 k 个最近邻样本
        :param row: 样本
        :return:
        near_sim_list: 同类的 k 个最近邻样本，内是样本的索引
        miss_sim_dict: 每个异类的 k 个最近邻样本，键是不同类的标签，值是 k 个最近邻样本的索引
        """
        '''同类的 k 个最近邻样本索引列表'''
        near_sim_list = list()
        '''每个异类的 k 个最近邻样本，键是不同类的标签，值是 k 个最近邻样本的索引列表'''
        miss_sim_dict = dict()
        '''样本标签'''
        row_label = row[self.label_name]
        '''遍历所有标签'''
        for label in self.label_set:
            '''同类标签与每个异类标签'''
            '''这里实现的是有监督的'''
            if label == row_label:
                '''同类用于加权的 k 个最近邻样本列表'''
                near_sim_list = self.distance(row, label)
            else:
                '''每个不同类用于加权的 k 个最近邻样本列表字典'''
                miss_sim_dict[label] = self.distance(row, label)
        return near_sim_list, miss_sim_dict

    def get_weight(self, feature: str, row: pandas.Series, near_hit: List[int],
                   near_miss: Dict[str, List[int]]) -> float:
        """
        返回一次迭代过程中一个特征的得分
        :param feature: 特征
        :param row: 当前被选中的样本
        :param near_hit: 与 row 同类用于加权的 k 个最近邻样本索引
        :param near_miss: 与 row 所有不同类用于加权的 k 个最近邻样本索引
        :return:
        weight: 一次迭代过程中一个特征的得分
        """
        '''抽样次数'''
        sample = int(len(self.sample_index))
        '''当前样本的标签'''
        label_r = row[-1]
        '''连续特征'''
        hit = 0
        miss = 0
        if self.attribute_dict[feature] == 0:
            '''连续特征，数据都已经归一化'''
            '''与 label_r 同类的用于加权的 k 个最近邻样本'''
            for index in near_hit:
                hit += abs(row[feature] - self.data.iloc[index][feature])
            '''与 label_r 所有不同类的用于加权的 k 个最近邻样本'''
            for label_c, k_list in near_miss.items():
                '''label_c 与 label_r 不同标签，有 len(label_list) - 1 类'''
                tmp_param = self.gain_near_miss_param(label_r, label_c)
                for k_index in k_list:
                    '''不同子类的系数不同'''
                    miss += tmp_param * abs(row[feature] - self.data.iloc[k_index][feature])
        else:
            '''离散特征，有监督算法可以使用 VDM 度量，无监督算法不太建议使用 VDM，因为没有标签'''
            '''为了统一起见，可以概率化离散特征值'''
            for index in near_hit:
                '''与 label_r 同类的用于加权的 k 个最近邻样本'''
                hit += (0 if row[feature] == self.data.iloc[index][feature] else 1)
            '''与 label_r 所有不同类的用于加权的 k 个最近邻样本'''
            for label_c, k_list in near_miss.items():
                '''label_c 与 label_r 不同标签，有 len(label_list) - 1 类'''
                tmp_param = self.gain_near_miss_param(label_r, label_c)
                for k_index in k_list:
                    '''不同子类的系数不同，加权使用非 0 即 1'''
                    miss += tmp_param * (0 if row[feature] == self.data.iloc[k_index][feature] else 1)
        '''一次迭代在一个特征上的得分'''
        weight = (hit - miss) / (sample * self.k)
        return weight

    def relief_f(self) -> Dict[str, float]:
        """
        过滤式特征选择算法 Relief-F
        :return:
        feature_weight: 在抽样样本上所有特征的权重
        """
        '''初始化部分参数，不同子类实现的方式不一样'''
        self.init_abstract_msg()
        '''特征权重列表，每一维记录着每一次迭代的特征权重'''
        score = list()
        for index in self.sample_index:
            '''一次抽样的每个特征权重'''
            one_score = dict()
            '''被选中的样本，是有标签的'''
            row = self.data.iloc[index]
            '''abstractmethod'''
            '''返回与 row 同标签用于加权的 k 个最近邻样本和不同标签用于加权的 k 个最近邻样本'''
            '''特别需要注意的是，有监督的 ReliefF 算法是在与当前同标签的样本集合中寻找 k '''
            near_hit_list, near_miss_dict = self.get_neighbor(row)
            '''遍历所有特征'''
            for feature in self.attribute_list:
                '''连续型特征和离散型特征的加权方式不一样'''
                weight = self.get_weight(feature, row, near_hit_list, near_miss_dict)
                '''返回这次迭代过程每个特征'''
                one_score[feature] = weight
            '''one_score 是一次迭代过程中的每个特征得分'''
            score.append(one_score)
        '''pandas.DataFrame 类型数据，展示了每一次迭代过程中每个特征的得分'''
        feature_weight = pandas.DataFrame(score)
        '''求和，得到每个特征的最后得分'''
        return feature_weight.sum()
