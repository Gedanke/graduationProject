# -*- coding: utf-8 -*-

import os
import abc
import pandas
from abc import ABC
from abc import ABCMeta, abstractmethod
from sklearn.neighbors import KDTree

folder = "/home/dfs/common/148_data"


class A(metaclass=ABCMeta):
    def __init__(self, a):
        self.a = a

    def AA(self):
        print("AA")

    def fun(self):
        print(self.a)

    @abstractmethod
    def common(self):
        print("abc.abstractmethod")


class B(A):
    def __init__(self, a):
        A.__init__(self, a)
        self.b = 0

    def fun(self):
        print(self.a)
        print(self.b)

    def BB(self):
        super().AA()
        print("BB")

    def common(self):
        print("abc.abstractmethod abc.abstractmethod")


# metaclass=ABCMeta
class IStream(metaclass=ABCMeta):
    @abc.abstractmethod
    def read(self, maxbytes=-1):
        print("read")

    @abstractmethod
    def write(self, data=""):
        print("wire")


class II(IStream):

    def read(self, maxbytes=-1):
        print("READ")

    # super().read()

    def write(self, data=""):
        super().write()
        print("WIRE")


if __name__ == "__main__":
    I = II()
    I.read()
    I.write()
