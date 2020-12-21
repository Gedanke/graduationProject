# -*- coding: utf-8 -*-


from core.dealData import *

"""
将 original_path 路径的txt文件，以 separator 为分割符，以 attribute_name 为列名(含标签)
使用 TransformData 类，得到与txt文件同一路径下的csv文件
"""
original_path = "../originalDataSet/wall-following/wall-following.txt"
separator = " "
attribute_name = ['US1', 'US2', 'US3', 'US4', 'US5', 'US6', 'US7', 'US8', 'US9', 'US10', 'US11', 'US12', 'US13', 'US14',
                  'US15', 'US16', 'US17', 'US18', 'US19', 'US20', 'US21', 'US22', 'US23', 'US24', 'label']


def fun1():
    """
    使用 TransformData 类，调用一次即可
    :return:
    """
    t = TransformData(original_path, separator, attribute_name)
    '''使用 mine_deal() 或者 standard_data() 方法都可'''
    t.mine_deal()


if __name__ == "__main__":
    ''''''
    fun1()
