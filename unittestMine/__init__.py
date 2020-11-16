# -*- coding: utf-8 -*-

import os
import unittest
from abc import ABCMeta, abstractmethod
from typing import List

import pandas

folder = "/home/dfs/common/148_data"

path = r"/home/dfs/文档/books/Mine/UCI_data/data/credit-approval/credit_approval.csv"


class ReliefF(metaclass=ABCMeta):
    def __init__(self, data: pandas.DataFrame, sample_rate: float):
        """
        :param data:
        :param sample_rate:
        """
        self.data = data
        self.rate = sample_rate


d = pandas.read_csv(path)
print(isinstance(d, pandas.DataFrame))
print(type(d))
print(type(["a", "b"]))
print(isinstance(["a", "b"], List))
