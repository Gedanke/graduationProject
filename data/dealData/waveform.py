# -*- coding: utf-8 -*-

from core.dealData import *

"""
将 original_path 路径的txt文件，以 separator 为分割符，以 attribute_name 为列名(含标签)
使用 TransformData 类，得到与txt文件同一路径下的csv文件
"""
original_path = "../originalDataSet/waveform/waveform.txt"
separator = " "
attribute_name = ['Attr1', 'Attr2', 'Attr3', 'Attr4', 'Attr5', 'Attr6', 'Attr7', 'Attr8', 'Attr9', 'Attr10', 'Attr11',
                  'Attr12', 'Attr13', 'Attr14', 'Attr15', 'Attr16', 'Attr17', 'Attr18', 'Attr19', 'Attr20', 'Attr21',
                  "Label"]

attribute_dict = {'Attr1': 0, 'Attr2': 0, 'Attr3': 0, 'Attr4': 0, 'Attr5': 0, 'Attr6': 0, 'Attr7': 0, 'Attr8': 0,
                  'Attr9': 0, 'Attr10': 0, 'Attr11': 0, 'Attr12': 0, 'Attr13': 0, 'Attr14': 0, 'Attr15': 0, 'Attr16': 0,
                  'Attr17': 0, 'Attr18': 0, 'Attr19': 0, 'Attr20': 0, 'Attr21': 0, "Label": -1}


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
    # fun1()
    # for i in range(21):
    #     attribute_name.append("Attr" + str(i + 1))
    #     attribute_dict["Attr" + str(i + 1)] = 0
    # print(attribute_name)
    # print(attribute_dict)
