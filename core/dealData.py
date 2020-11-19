# -*- coding: utf-8 -*-


import os
import csv
import pandas
from typing import List, Dict

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

    def gain_extension(self):
        """
        拆分文件路径 self.path
        :return:
        @file_path : 返回文件路径
        @shot_name : 返回文件名
        @extension : 返回文件后缀
        """
        file_path, temp_filename = os.path.split(self.path)
        shot_name, extension = os.path.splitext(temp_filename)
        return file_path, shot_name, extension

    def mine_deal(self):
        """
        根据 path，separator，attribute_name 将 txt 数据集转换为 csv 格式
        这里的方法是从文本数据集中一行行读取并写入，虽不及 standard_data() 方法，但是其可以手动处理一些情况
        如，可以处理每一行数据除了 separator 外其他的无关字符
        :return:
        """
        _path, shot_name, extension = self.gain_extension()
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
        _path, shot_name, extension = self.gain_extension()
        '''生成的 cvs 文件的完整路径'''
        global csv_path
        csv_path = _path + "/" + shot_name + ".csv"
        data.to_csv(csv_path, index=0)
        self.path = csv_path


class DealData(object):
    """
    对 csv 文件处理后得到两份完整的
    处理好后的csv文件
    对连续特征归一化，对离散特征可以参与数值运算，如果能直接被数值化，其也需要和连续特征一样被归一化，内容统一为浮点数
    之后以 remove_rate 去除数据集中的标签
    一份的标签保持原样，在原来的文件名后加 Supervised
    另一份以 remove_rate 去除标签，在原来的文件名后加 UnSupervised
    两份文件除了有无标签，其他样本的行索引必须完全一致
    也就是两份数据集除了一份样本缺失了大部分标签外，其他的内容完全一样
    将这两份数据集保存在父父文件的同一路径下的 processedDataSet 文件夹里
    
    """

    def __init__(self, attribute_dict: Dict[str, int], remove_rate: float):
        """
        :param attribute_dict: 
        attribute_dict 是属性名字典，如果一个属性是连续特征，则值为 0，若为离散特征，则为 1，若为标签，则为 -1
        这里对于离散特征，不区分有序离散特征与无序离散特征，只要是离散特征就为 1
        属性名字典按先特征，最后标签的顺序
        :param remove_rate: 
        以 remove_rate 比例除去数据集中的标签
        """
        self.path = csv_path
        self.attribute_dict = attribute_dict
        self.remove_rate = remove_rate


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
