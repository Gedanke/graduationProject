# -*- coding: utf-8 -*-

import pandas
import operator
from typing import List
from core.dealData import *

original_path = "../originalDataSet/abalone/abalone.txt"
separator = " "
attribute_name = [
    "Label", "Length", "Diam", "Height", "Whole", "Shucked", "Viscera", "Shell", "Rings"
]


def fun1():
    t = TransformData(original_path, separator, attribute_name)
    t.mine_deal()


csv_path = "../originalDataSet/abalone/abalone.csv"
attribute_dict = {
    "Length": 0, "Diam": 0, "Height": 0, "Whole": 0, "Shucked": 0, "Viscera": 0, "Shell": 0, "Rings": 0,
    "Label": -1
}
remove_rate = 0.8


def fun2():
    d = DealData(csv_path, attribute_dict, remove_rate)
    d.deal_data()


data_path = ""
divide_rate = 1

if __name__ == "__main__":
    # fun1()
    # print(csv_path)
    fun2()
