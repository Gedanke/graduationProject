# -*- coding: utf-8 -*-

import os
import abc
from abc import ABC
from abc import ABCMeta, abstractmethod

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


if __name__ == "__main__":
    _b = B(1)
    _b.BB()
    _b.fun()
    _b.common()
