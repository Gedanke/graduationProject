# -*- coding: utf-8 -*-

from .basic import *
import pandas

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


# def distance_measure(current, aims) -> float:
#     return 0


class ReliefFSupervised(ReliefF):
    """
    有监督下的 ReliefF 的算法
    样本空间中的所有样本都有标签
    当前被选中的样本，会在与其同标签的样本集中寻找最近邻的 k 个样本
    之后在与其不同标签的样本集中都去寻找最近邻的 k 个样本

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
        ReliefF.__init__(self, data, attribute_dict, sample_rate, k)

    def init_abstract_msg(self):
        """
        初始化相关参数，有监督下的 ReliefF 的算法的实现
        self.sample_index: 抽样样本索引列表
        self.label_set: 标签集合
        self.search_space: 近邻搜索空间，键为标签，值为样本索引
        self.label_nums: 每个标签的数量
        :return:
        """
        '''有监督的在全体样本上抽取，所有样本都有标签，延用 ReliefF 中的方法'''
        '''self.sample_index，采样索引'''
        '''self.label_set，标签集合'''
        '''self.search_space，按标签分类，在自己标签上进行搜索，键为标签，值为样本索引'''
        '''self.label_nums，每个标签的数量，便于 self.gain_near_miss_param 的计算'''
        super().init_abstract_msg()

    def gain_near_miss_param(self, label_r: str, label_c: str) -> float:
        """
        不同类待乘的系数，有监督下的 ReliefF 的算法的实现是
        该类占所有不同类的比例
        返回与 label_r 不同标签的 label_c 特征加权时的系数
        :param label_r: 当前样本的标签
        :param label_c: 与 label_r 不同标签样本的标签 label_c
        :return:
        @param: 待乘的系数
        """
        param = self.label_nums[label_c] / (self.sample_num - self.label_nums[label_r])
        return param

    def distance(self, row: pandas.Series, label: str) -> List[int]:
        """
        在当前样本 row 搜索空间里寻找最近邻居样本，有监督下的 ReliefF 的算法的实现是
        row 在指定的 label 下的样本空间里搜索 k 个最近邻样本
        :param row: 当前样本
        :param label: 在该 label 下搜索 k 个最近邻样本
        :return:
        sim: k 个最近邻样本的索引
        """
        '''有监督下的 ReliefF 的算法的实现'''
        sim = list()
        ''''''

        ''''''

        return sim


class ReliefFUnsupervised(ReliefF):
    """
    无监督下的 ReliefF 的算法
    样本空间中的样本只有部分含有标签
    当前被选中的样本，会选择一个与当前样本同标签的最近邻样本，然后会在所有的无标签样本集中寻找最近邻的 k 个样本
    之后在与其不同标签的样本集中都去寻找一个最近邻样本，这些所有不同类的最近邻样本都会在所有的无标签样本集中寻找最近邻的 k 个样本

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
        ReliefF.__init__(self, data, attribute_dict, sample_rate, k)

    def init_abstract_msg(self):
        """
        初始化相关参数，无监督下的 ReliefF 的算法的实现
        self.sample_index: 抽样样本索引列表
        self.label_set: 标签集合
        self.search_space: 近邻搜索空间，键为标签，值为样本索引
        self.label_nums: 每个标签的数量
        :return:
        """
        super().init_abstract_msg()
        '''self.sample_index'''
        '''无监督的只在有标签的样本上抽取，无标签的样本标签名处为 None，只需要用总数减去 None 的数目'''
        sample = int(round((self.sample_num - self.label_nums["None"]) * self.sample_rate))
        self.sample_index = random.sample(list(range(self.sample_num)), sample)
        '''self.label_set，无监督算法标签集合需要去除 None'''
        self.label_set.remove("None")
        '''self.search_space，所有标签都在没有标签的样本上搜索，键为 Any，表示所有标签，值为样本索引，索引建议排序'''
        '''self.label_nums，每个标签的数量，便于 self.gain_near_miss_param 的计算，无监督下的系数可能不需要'''
        '''原有的 self.search_space 弹出键值为 None 的索引列表，弹出了无标签数据集索引'''
        no_label_list = self.search_space.pop("None")
        '''此时的 self.search_space 是不含无标签数据集的索引，需要拷贝一份'''
        self.label_nums = self.search_space.copy()
        '''self.search_space 清空后，再添加无标签数据集的全部索引'''
        self.search_space.clear()
        self.search_space["Any"] = no_label_list

    def gain_near_miss_param(self, label_r: str, label_c: str) -> float:
        """
        不同类待乘的系数，无监督下的 ReliefF 的算法的实现是
        有标签数据集中该类占所有不同类的比例或者 1 / (len(label_set) - 1)
        返回与 label_r 不同标签的 label_c 特征加权时的系数
        :param label_r: 当前样本的标签
        :param label_c: 与 label_r 不同标签样本的标签 label_c
        :return:
        @param: 待乘的系数
        """
        param = 1 / (len(self.label_set) - 1)
        return param

    def distance(self, row: pandas.Series, label: str) -> List[int]:
        """
        在当前样本 row 搜索空间里寻找最近邻居样本，无监督下的 ReliefF 的算法的实现是
        所 row 的标签和 label 相同，在 label 的搜索空间里找一个最近邻样本，
        该最近邻样本在所有无标签样本空间上寻找 k 个最近邻样本
        所 row 的标签和 label 不相同，在所有与 row 不同 label 的搜索空间里都找一个最近邻样本，
        该最近邻样本在所有无标签样本空间上寻找 k 个最近邻样本
        :param row: 当前样本
        :param label: 在该 label 下搜索 k 个最近邻样本
        :return:
        sim: k 个最近邻样本的索引
        """
        '''有监督，无监督，改进无监督搜索用于加权的 k 个最近邻样本略有不同'''
        sim = list()
        return sim


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
        ReliefF.__init__(self, data, attribute_dict, sample_rate, k)

    def init_abstract_msg(self):
        """
        初始化相关参数，改进后无监督下的 ReliefF 的算法是
        有标签数据集中该类占所有不同类的比例或者 1 / (len(label_set) - 1)
        self.sample_index: 抽样样本索引列表
        self.label_set: 标签集合
        self.search_space: 近邻搜索空间，键为标签，值为样本索引
        self.label_nums: 每个标签的数量
        :return:
        """
        super().init_abstract_msg()
        '''self.sample_index'''
        '''无监督的只在有标签的样本上抽取，无标签的样本标签名处为 None，只需要用总数减去 None 的数目'''
        sample = int(round((self.sample_num - self.label_nums["None"]) * self.sample_rate))
        self.sample_index = random.sample(list(range(self.sample_num)), sample)
        '''self.label_set，无监督算法标签集合需要去除 None'''
        self.label_set.remove("None")
        '''self.search_space，所有标签都在没有标签的样本与自己同标签的样本集合上搜索，键为标签，值为样本索引，索引建议排序'''
        '''self.label_nums，每个标签的数量，便于 self.gain_near_miss_param 的计算，无监督下的系数可能不需要'''
        '''原有的 self.search_space 弹出键值为 None 的索引列表，弹出了无标签数据索引'''
        no_label_list = self.search_space.pop("None")
        '''self.search_space 只含有有标签样本索引'''
        for label, value in self.search_space.items():
            '''每个有标签样本索引集合加上所有无标签数据索引'''
            self.search_space[label].extend(no_label_list)
        '''self.label_nums 去除无标签数据的数量'''
        self.label_nums.pop("None")

    def gain_near_miss_param(self, label_r: str, label_c: str) -> float:
        """
        不同类待乘的系数，改进无监督下的 ReliefF 的算法是
        有标签数据集中该类占所有不同类的比例或者 1 / (len(label_set) - 1)
        返回与 label_r 不同标签的 label_c 特征加权时的系数
        :param label_r: 当前样本的标签
        :param label_c: 与 label_r 不同标签样本的标签 label_c
        :return:
        @param: 待乘的系数
        """
        param = 1 / (len(self.label_set) - 1)
        return param

    def distance(self, row: pandas.Series, label: str) -> List[int]:
        """
        在当前样本 row 搜索空间里寻找最近邻居样本，改进无监督下的 ReliefF 的算法是
        所 row 的标签和 label 相同，在 label 的搜索空间里找一个最近邻样本，
        该最近邻样本在所有无标签样本空间与含 label 的有标签样本空间的并集上寻找 k 个最近邻样本
        所 row 的标签和 label 不相同，在所有与 row 不同 label 的搜索空间里都找一个最近邻样本，
        该最近邻样本在所有无标签样本空间与含该 label 的有标签样本空间的并集上寻找 k 个最近邻样本
        :param row: 当前样本
        :param label: 在该 label 下搜索 k 个最近邻样本
        :return:
        sim: k 个最近邻样本的索引
        """
        '''有监督，无监督，改进无监督搜索用于加权的 k 个最近邻样本略有不同'''
        sim = list()
        return sim
