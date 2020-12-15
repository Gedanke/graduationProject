# -*- coding: utf-8 -*-

import numpy
import pandas
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import sys

"""
该部分的任务是将大的数据集划分为若干块，从每一块中按一定比例抽取样本，汇总得到一个新的样本集合
读取的 pandas.DataFrame 类型数据，其中数据集已经经过清洗，离散特征熵值连续化，所有特征都已经归一化
处理后，数据集会被划分到不同块中
会从每块中按一定的比例抽取数据集

KDTree：
使用 Kd-Tree 算法将数据划分到叶子节点中
https://blog.csdn.net/qq_33690156/article/details/52452950
https://www.pythonf.cn/read/79091

KMeans:
使用 K-Means 算法将数据集按标签类别数进行划分
https://www.cnblogs.com/pinard/p/6164214.html

"""


class KDTree(object):
    """
    Kd-Tree 算法
    首先在方差最大的特征维度上划分数据集，之后在剩下的特征中选择方差大的，以此类推
    最后所有的样本都被划分到叶子节点中，当然一个叶子节点会含有多个样本
    在每个叶子节点中按比例抽取样本

    """

    def __init__(self, data: pandas.DataFrame):
        """
        :param data:
        数据集 pandas.DataFrame 或 pandas.core.frame.DataFrame 类型
        """
        self.data = data


def distance(vecA, vecB):
    """
    计算距离的方法
    :param vecA: 向量 A
    :param vecB: 向量 B
    :return:
    返回两个向量的欧式距离
    """
    return numpy.sqrt(numpy.sum(numpy.power(vecA - vecB, 2)))


def rand_cent(data, k: int) -> numpy.mat:
    """
    K-Means 初始化质心的位置的方法，构建一个包含 k 个随机质心的集合
    :param data: 数据集
    :param k: 质心个数
    :return:
    centroids: 初始化得到的 k 个质心向量
    """
    '''获取样本的维度'''
    n = numpy.shape(data)[1]
    '''初始化一个 (k,n) 的全零矩阵，也就是初始聚类中心'''
    centroids = numpy.mat(numpy.zeros((k, n)))
    '''遍历矩阵的每一维度'''
    for index in range(n):
        '''获取矩阵该列的最小值'''
        min_index = numpy.min(data[:, index])
        '''获取矩阵该列的最大值'''
        max_index = numpy.max(data[:, index])
        '''得到矩阵该列的范围 (最大值 - 最小值)'''
        range_index = float(max_index - min_index)
        '''k 个质心向量的第 index 维数据值随机为位于 (最小值，最大值) 内的某一值'''
        '''random.rand(行，列) 产生这个形状的矩阵，且每个元素 in [0,1)'''
        centroids[:, index] = min_index + range_index * numpy.random.rand(k, 1)
    '''返回初始化得到的 k 个质心向量'''
    return centroids


def rand_cent_pluses(data, k: int) -> numpy.mat:
    """
    K-Means++ 初始化质心的位置的方法，构建一个包含 k 个质心的集合
    :param data: 数据集
    :param k: 质心个数
    :return:
    centroids: 初始化得到的 k 个质心向量
    """
    '''得到数据样本的维度'''
    n = numpy.shape(data)[1]
    '''得到数据样本的维度'''
    m = numpy.shape(data)[0]
    '''初始化为一个 (k,n) 的全零矩阵'''
    centroids = numpy.mat(numpy.zeros((k, n)))
    '''step1: 随机选择样本里的一个点'''
    centroids[0, :] = data[numpy.random.randint(m), :]
    # plot_center(data, numpy.array(centroids))
    '''迭代'''
    for c_id in range(k - 1):
        dist = list()
        '''遍历所有点'''
        for i in range(m):
            point = data[i, :]
            d = sys.maxsize
            '''扫描所有质心，选出该样本点与最近的类中心的距离'''
            for j in range(centroids.shape[0]):
                tmp_dist = distance(point, centroids[j, :])
                d = min(d, tmp_dist)
            dist.append(d)
        dist = numpy.array(dist)
        '''返回的是 dist 里面最大值的下标，对应的是上面循环中的 i'''
        next_centroid = data[numpy.argmax(dist), :]
        '''选出了下一次的聚类中心，开始 k+1 循环'''
        centroids[c_id + 1, :] = next_centroid
        # plot_center(data, numpy.array(centroids))
    return centroids


def plot_center(data: numpy.mat, centroids: numpy.ndarray):
    """
    绘制质心
    :param data: 数据集
    :param centroids: 质心向量
    :return:
    """
    plt.scatter(data[:, 0], data[:, 1], marker=".",
                color="gray", label="data points")
    plt.scatter(centroids[:-1, 0], centroids[:-1, 1],
                color="black", label="previously selected centroids")
    plt.scatter(centroids[-1, 0], centroids[-1, 1],
                color="red", label="next centroid")
    plt.title('Select % d th centroid' % (centroids.shape[0]))
    plt.legend()
    plt.xlim(-5, 12)
    plt.ylim(-10, 15)
    plt.show()


class MKMeans(object):
    """
    只针对半监督数据集
    在选取原始的 k 个基本点时，先将数据集中含有标签的样本按类别划分成不同集合
    在每一个类别集合中，选取该类所有样本的中心点为该类别的初始点
    最后每个类别都有一个初始基本点，之后进行聚类，k 类
    在每类中按比例抽取样本
    指定聚类的类别 k，初始中心点
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    https://zhuanlan.zhihu.com/p/149441104

    """

    def __init__(self, data: pandas.DataFrame, parameter_dict: Dict[str, Any] = None):
        """
        :param data:
        数据集 pandas.DataFrame 或 pandas.core.frame.DataFrame 类型
        :param parameter_dict:
        其他的参数字典，可以考虑的有最大迭代次数
        """
        self.data = data
        self.parameter_dict = parameter_dict
        '''将 self.data 的数据域(只含特征值，不含标签)转换为 numpy 矩阵'''
        self.array_data = None
        '''k 值是有标签的类别数'''
        self.k = 0
        '''中心点，是相同标签点的中心点'''
        self.center_point = None
        '''属性列表，包含标签名'''
        self.attribute_list = list(self.data.columns)
        '''一些默认参数，与 sklearn.cluster.KMeans 类相似，如果想改，通过 parameter_t 传入'''
        '''初始化部分参数'''
        self.init_data()

    def init_data(self):
        """
        初始化部分参数
        self.k self.center_point self.array_data
        :return:
        """
        '''将数据集按 self.label_name 分组，键为标签，值为相同标签样本的索引，已经是字典序了'''
        label_groups = self.data.groupby(self.attribute_list[-1]).groups
        '''获取标签数量，即聚类的类别数'''
        self.k = len(label_groups) - 1
        '''不在 __init__ 中使其为 list()，是因为之后的 k_means_m 方法中的
        dist_juli = distance(centroids[j, :], self.array_data[i, :])
        的 centroids 会因为 self.center_point 的初始类型 list() 发出警告，当然不影响结果'''
        self.center_point = list()
        for label, value in label_groups.items():
            '''求得每个标签下的所有标签距离的均值'''
            if label != "None":
                self.center_point.append(
                    list(self.data.iloc[list(value)].mean()))
        '''inplace=False，删除标签所在列不改变原数据，返回一个执行删除操作后的新 dataframe，将其转换为 numpy 矩阵'''
        self.array_data = numpy.mat(self.data.drop(
            columns=self.attribute_list[-1], inplace=False))
        '''self.center_point 转换为 numpy.mat 类型'''
        self.center_point: numpy.mat = numpy.mat(self.center_point)

    def k_means_basic(self, centroids_: numpy.mat) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        基本的 K-Means 聚类算法
        :param centroids_: 初始聚类中心
        :return:
        centroids: k 个类质心的位置坐标
        cluster_class: 样本所处的类以及到该类质心的距离
        """
        '''获取矩阵的行数'''
        m = numpy.shape(self.array_data)[0]
        '''初始化一个 (m,2) 的全零矩阵，用来记录每一个样本所属类，距离类中心的距离'''
        cluster_class = numpy.mat(numpy.zeros((m, 2)))
        '''创建初始的 k 个质心向量'''
        centroids = centroids_
        '''聚类结果是否发生变化的布尔类型'''
        cluster_changed = True
        '''终止条件，所有数据点聚类结果不发生变化'''
        while cluster_changed:
            '''聚类结果变化布尔类型置为 False'''
            cluster_changed = False
            '''遍历矩阵的每一个向量'''
            for i in range(m):
                '''初始化最小距离为正无穷大'''
                min_dist = float('inf')
                '''初始化最小距离对应的索引为 -1'''
                min_index = -1
                '''循环 self.k 个类的质心'''
                for j in range(self.k):
                    '''计算每个样本到质心的欧式距离'''
                    dist_juli = distance(
                        centroids[j, :], self.array_data[i, :])
                    '''如果当前距离小于当前最小距离'''
                    if dist_juli < min_dist:
                        '''当前距离为最小距离，最小距离对应的索引为 j(第j类)'''
                        min_dist = dist_juli
                        min_index = j
                '''当前聚类结果第 i 个样本的聚类结果发生变化，cluster_changed 置为 True，继续进行聚类算法'''
                if cluster_class[i, 0] != min_index:
                    cluster_changed = True
                '''更新当前变化样本的聚类结果和误差平方'''
                cluster_class[i, :] = min_index, min_dist ** 2
            '''遍历每一个质心'''
            for cent in range(self.k):
                '''将矩阵中所有属于当前质心类的样本通过条件过滤筛选出来'''
                pst_cluster = self.array_data[numpy.nonzero(
                    cluster_class[:, 0].A == cent)[0]]
                '''计算这些数据的均值(axis=0:求均值)，作为该类质心向量'''
                centroids[cent, :] = numpy.mean(pst_cluster, axis=0)
        '''返回 k 个聚类，聚类结果以及误差'''
        return centroids, cluster_class

    def k_means(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        KMeans 聚类算法
        :param dist_means:
        :param create_cent:
        :return:
        centroids: k 个类质心的位置坐标
        cluster_class: 样本所处的类以及到该类质心的距离
        """
        '''聚类中心由 KMeans 聚类算法给出'''
        centroids_ = rand_cent(self.array_data, self.k)
        centroids, cluster_class = self.k_means_basic(centroids_)
        return centroids, cluster_class

    def k_means_pluses(self):
        """
        KMeans++ 聚类算法
        https://zhuanlan.zhihu.com/p/149978127
        :return:
        centroids: k 个类质心的位置坐标
        cluster_class: 样本所处的类以及到该类质心的距离
        """
        '''聚类中心由 KMeans++ 聚类算法给出'''
        centroids_ = rand_cent_pluses(self.array_data, self.k)
        centroids, cluster_class = self.k_means_basic(centroids_)
        return centroids, cluster_class

    def k_means_m(self):
        """
        KMeans 改进初始聚类中心的聚类算法
        :return:
        centroids: k 个类质心的位置坐标
        cluster_class: 样本所处的类以及到该类质心的距离
        """
        '''聚类中心由改进的 KMeans 聚类算法给出'''
        centroids, cluster_class = self.k_means_basic(self.center_point)
        return centroids, cluster_class
