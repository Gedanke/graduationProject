# -*- coding: utf-8 -*-

import pandas
import operator
from typing import List
from core.dealData import *

original_path = "../originalDataSet/abalone/abalone.txt"
separator = " "
attribute_name = ["D", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]


def fun1():
    t = TransformData(original_path, separator, attribute_name)
    t.mine_deal()


csv_path = "../originalDataSet/abalone/abalone.csv"

attribute_dict = {
    "A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0, "A6": 0, "A7": 0, "A8": 1,
    "D": -1
}
remove_rate = 0.2

data_path = ""
divide_rate = 1

if __name__ == "__main__":
    fun1()
    print(csv_path)
