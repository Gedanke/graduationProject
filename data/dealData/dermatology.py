# -*- coding: utf-8 -*-


from core.dealData import *

"""
将 original_path 路径的txt文件，以 separator 为分割符，以 attribute_name 为列名(含标签)
使用 TransformData 类，得到与txt文件同一路径下的csv文件
"""
original_path = "../originalDataSet/dermatology/dermatology.txt"
separator = ","
attribute_name = ['Attr1', 'Attr2', 'Attr3', 'Attr4', 'Attr5', 'Attr6', 'Attr7', 'Attr8', 'Attr9', 'Attr10', 'Attr11',
                  'Attr12', 'Attr13', 'Attr14', 'Attr15', 'Attr16', 'Attr17', 'Attr18', 'Attr19', 'Attr20', 'Attr21',
                  'Attr22', 'Attr23', 'Attr24', 'Attr25', 'Attr26', 'Attr27', 'Attr28', 'Attr29', 'Attr30', 'Attr31',
                  'Attr32', 'Attr33', 'Attr34', 'label']


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
