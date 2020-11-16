# -*- coding: utf-8 -*-


import os
import pandas
from typing import List, Dict

"""
TransformData：
根据 txt 文件和文件内容分割符，手动增加属性名，将 txt 文件转换为同一路径下的 csv 文件
得到完整的 csv 文件，编码为 utf-8
在 csv 文件中
第一行是属性列名，最后一列是标签，其余列均为特征
内容统一为浮点数，字符串，不能有缺失的属性值，为完备数据集
注：
这里只是单纯得到 csv 文件，去除标签等其他操作放到下一步

DealData:
对处理好的 csv 文件进行进一步处理
将连续特征归一化，对离散特征的各个属性求得其熵值，之后也对其进行归一化
得到两份完整的，处理好后的csv文件
一份的标签保持原样，另一份以 remove_rate 去除标签
将这两份数据集保存在当前文件的子文件夹里

DivideData:
以一定 divide_rate 比例划分数据为训练集和测试集
为 1，不用划分

"""

csv_path = ""


class TransformData(object):
    def __init__(self, path: str, separator: str, attribute_name: List[str]):
        """
        :param path: txt 文件的路径
        :param separator: 数据之间的分割符
        :param attribute_name: 数据集每列的列名列表
        """
        self.path = path
        self.separator = separator
        self.attribute_name = attribute_name


class DealData(object):
    def __init__(self, attribute_dict: Dict[str, int], remove_rate: float):
        """
        :param attribute_dict:
        :param remove_rate:
        """
        self.path = csv_path
        self.attribute_dict = attribute_dict
        self.remove_rate = remove_rate


class DivideData(object):
    def __int__(self, data_path, divide_rate: float):
        """
        :param data_path
        :param divide_rate:
        """
        self.path = data_path
        self.divide_rate = divide_rate
