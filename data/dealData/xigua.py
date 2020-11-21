# -*- coding: utf-8 -*-

import pandas
import operator
from typing import List
from core.dealData import *

original_path = "../originalDataSet/testData/xigua.txt"
separator = ","
attribute_name = [
    "色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率", "好瓜"
]


def fun1():
    t = TransformData(original_path, separator, attribute_name)
    t.mine_deal()


csv_path = "../originalDataSet/testData/xigua.csv"
attribute_dict = {
    "色泽": 1, "根蒂": 1, "敲声": 1, "纹理": 1, "脐部": 1, "触感": 1, "密度": 0, "含糖率": 0,
    "好瓜": -1
}
remove_rate = 0.8


def fun2():
    d = DealData(csv_path, attribute_dict, remove_rate)
    d.variance_data()
    # print(d.attribute_variance)
    d.deal_data()



data_path = ""
divide_rate = 1

if __name__ == "__main__":
    # fun1()
    # print(csv_path)
    fun2()

