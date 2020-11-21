# -*- coding: utf-8 -*-

import os
import csv
import random
import pandas
import operator
from typing import List, Dict
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

"""
TransformData：
根据 txt 文件和文件内容分割符，手动增加属性名，将 txt 文件转换为同一路径下的 csv 文件
得到完整的 csv 文件，编码为 utf-8
注：
这里只是单纯得到 csv 文件，去除标签等其他操作放到下一步


DealData:
对处理好的 csv 文件进行进一步处理
将连续特征归一化，对离散特征的各个属性求得其熵值，之后也对其进行归一化
得到处理好后的 csv 文件


DivideData:
以一定 divide_rate 比例划分数据为训练集和测试集
为 1，不用划分，格式依然是 csv

"""

csv_path = ""


def gain_extension(path):
    """
    拆分文件路径 path
    :return:
    @file_path : 返回文件路径
    @shot_name : 返回文件名
    @extension : 返回文件后缀
    """
    file_path, temp_filename = os.path.split(path)
    shot_name, extension = os.path.splitext(temp_filename)
    return file_path, shot_name, extension


class TransformData(object):
    """
    对 txt 文本进行处理，得到符合要求的 csv 文件
    在 csv 文件中
    第一行是属性列名，最后一列是标签，其余列均为特征
    内容统一为浮点数，字符串，不能有缺失的属性值，为完备数据集
    处理后的 csv 文件和 txt 文件同名，放在同一路径下
    
    """

    def __init__(self, path: str, separator: str, attribute_name: List[str]):
        """
        :param path: 
        txt 文件的路径
        :param separator: 数据之间的分割符
        :param attribute_name: 数据集每列的列名列表
        """
        self.path = path
        self.separator = separator
        self.attribute_name = attribute_name

    def mine_deal(self):
        """
        根据 path，separator，attribute_name 将 txt 数据集转换为 csv 格式
        这里的方法是从文本数据集中一行行读取并写入，虽不及 standard_data() 方法，但是其可以手动处理一些情况
        如，可以处理每一行数据除了 separator 外其他的无关字符
        :return:
        """
        _path, shot_name, extension = gain_extension(self.path)
        '''生成的 cvs 文件的完整路径'''
        global csv_path
        csv_path = _path + "/" + shot_name + ".csv"
        new_file = open(csv_path, "w+", newline='')
        writer = csv.writer(new_file)
        '''先将列名写入'''
        writer.writerow(self.attribute_name)
        data = open(self.path)
        lines = data.readlines()
        for index in range(len(lines)):
            lines[index] = lines[index].strip('\n')
            '''如果每行还有其他待去除的，再次使用 strip() 函数即可'''
            # lines[index]=lines[index].strip()
            line = lines[index].split(self.separator)
            writer.writerow(line)
        data.close()
        new_file.close()
        self.path = csv_path

    def standard_data(self):
        """
        根据 path，separator，attribute_name 将 txt 数据集转换为 csv 格式
        使用 pandas 自带的方法，快速便捷
        :return:
        """
        data = pandas.read_table(
            self.path, sep=self.separator, names=self.attribute_name
        )
        _path, shot_name, extension = gain_extension(self.path)
        '''生成的 cvs 文件的完整路径'''
        global csv_path
        csv_path = _path + "/" + shot_name + ".csv"
        data.to_csv(csv_path, index=0)
        self.path = csv_path


class DealData(object):
    """
    对 csv 文件处理后得到两份完整的处理好后的csv文件
    对连续特征归一化，对离散特征可以参与数值运算，
    如果能直接被数值化，其也需要和连续特征一样被归一化，内容统一为浮点数，
    如果不能，连续特征也需要归一化，离散特征使用 VDM 度量
    此时需要求每个特征的方差，连续特征归一化后求其方差，离散特征概率化后求其方差
    之后以 remove_rate 去除数据集中的标签
    一份的标签保持原样，在原来的文件名后加 Supervised
    另一份以 remove_rate 去除标签，在原来的文件名后加 UnSupervised
    两份文件除了有无标签，其他样本的行索引必须完全一致
    也就是两份数据集除了一份样本缺失了大部分标签外，其他的内容完全一样
    将这两份数据集保存在父父文件的同一路径下的 processedDataSet 文件夹里
    
    """

    def __init__(self, path: str, attribute_dict: Dict[str, int], remove_rate: float):
        """
        :parameter path:
        处理好后的 csv 文件的路径
        :param attribute_dict: 
        attribute_dict 是属性名字典，如果一个属性是连续特征，则值为 0，若为离散特征，则为 1，若为标签，则为 -1
        这里对于离散特征，不区分有序离散特征与无序离散特征，只要是离散特征就为 1
        属性名字典按先特征，最后标签的顺序
        :param remove_rate: 
        以 remove_rate 比例除去数据集中的标签
        """
        self.path = path
        self.attribute_dict = attribute_dict
        self.remove_rate = remove_rate
        '''从 csv_path 中读取的处理好后的数据'''
        self.data = pandas.read_csv(self.path)
        '''连续特征列表'''
        self.attribute_continuous = list()
        '''所有特征的方差，键和 attribute_dict 一样'''
        self.attribute_variance = dict()
        '''样本数量'''
        self.sample_num = int(len(self.data))
        ''''''
        self.label = ""
        ''''''
        ''''''
        self.supervised_path = ""
        self.unSupervised_path = ""
        self.init_data()

    def init_data(self):
        """
        类成员变量初始化
        :return:
        """
        for key, value in self.attribute_dict.items():
            if value != -1:
                '''初始化方差字典'''
                self.attribute_variance[key] = 0
                if value == 0:
                    '''收集连续特征的列表'''
                    self.attribute_continuous.append(key)
            else:
                self.label = key
        path_, shot_name, extension = gain_extension(self.path)
        path_list = path_.split("/")
        path_list[1] = "processedDataSet"
        for path_l in path_list:
            self.supervised_path += path_l + "/"
        self.unSupervised_path = self.supervised_path
        self.supervised_path += shot_name + "_supervised" + extension
        self.unSupervised_path += shot_name + "_unSupervised" + extension

    def variance_data(self):
        """
        将 self.data 进行归一化处理
        连续特征归一化处理，离散特征概率化
        使用 DataFrame.sample 函数：
        DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
        n 是选取的条数，frac 是选取的比例，replace 是可不可以重复选
        weights 是权重，random_state 是随机种子，axis 为 0 是选取行，为 1 是选取列。
        :return:
        """
        model = MinMaxScaler()
        if len(self.attribute_continuous) != 0:
            '''归一化连续特征'''
            self.data[self.attribute_continuous] = model.fit_transform(self.data[self.attribute_continuous])
        ''''''
        self.data.to_csv(self.path, index=False, sep=",")
        self.data.to_csv(self.supervised_path, index=False, sep=",")
        '''求连续和离散特征方差特征'''
        for key, value in self.attribute_dict.items():
            # 连续特征
            '''https://github.com/pandas-dev/pandas/issues/1798'''
            if value == 0:
                '''此种方式计算的方差是无偏的，分母是 n-1'''
                # print(self.data[key].var())
                '''此种方式计算的方差是有偏的，分母是 n，与 numpy 计算的方差一样'''
                # print(self.data[key].values.var())
                self.attribute_variance[key] = self.data[key].values.var()
            # 连续特征
            elif value == 1:
                '''统计每个属性出现的次数'''
                d = dict(Counter(self.data[key]).items())
                s = 0
                avg = 1 / len(d)
                '''计算方差'''
                for key_, value_ in d.items():
                    s += pow(abs(value_ / self.sample_num - avg), 2)
                self.attribute_variance[key] = s / len(d)

    def deal_data(self):
        """
        以 remove_rate 去除数据集中的标签
        一份数据集保持原样，另一份去除标签，两者区别仅仅在于部分样本是否存在方差
        :return:
        """
        ''''''
        sample = int(round(int(self.sample_num * self.remove_rate)))
        ''''''
        sample_index = sorted(random.sample(list(range(self.sample_num)), sample))
        for index in sample_index:
            ''''''
            self.data.loc[index, self.label] = "None"
        self.data.to_csv(self.unSupervised_path, index=False, sep=",")


class DivideData(object):
    """
    以 divide_rate 划分数据集，结束后
    训练集在文件后加 _train，测试集后加 _test
    若 divide_rate 为1，则没有被划分，保持原文件不变

    """

    def __int__(self, data_path, divide_rate: float):
        """
        :param data_path
        :param divide_rate:
        """
        self.path = data_path
        self.divide_rate = divide_rate
