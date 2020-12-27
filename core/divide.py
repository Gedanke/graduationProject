# -*- coding: utf-8 -*-

import numpy
import time
from typing import Any, Tuple
import matplotlib.pyplot as plt
import sys
from core.dealData import *

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


def deal_path(path):
    """
    :param path: 文件路径
    :return:
    new_path: 在该路径下，先创建一个 KMeans 文件夹，然后返回文件路径
    """
    file_path, shot_name, extension = gain_extension(path)
    '''先判断该路径是否存在'''
    folder = os.path.exists(file_path + "/kMeans/")
    '''不存在，可以创建该 KMeans 文件夹'''
    if not folder:
        os.makedirs(file_path + "/kMeans/")
    new_path = file_path + "/kMeans/" + shot_name + ".csv"
    return new_path


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
        dist = []
        '''选出了下一次的聚类中心，开始 k+1 循环'''
        centroids[c_id + 1, :] = next_centroid
    return centroids


def plot_center(data: numpy.mat, centroids: numpy.ndarray):
    """
    二维数据集绘制质心
    :param data: 数据集，只适用于二维数据集
    如果想在多维数据上使用，可以按特征类型，离散特征和连续特征对数据划分成二维，进行可视化
    :param centroids: 质心向量
    :return:
    """
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(), marker=".",
                color="gray", label="data points")
    plt.scatter(centroids[:-1, 0], centroids[:-1, 1],
                color="black", label="previously selected centroids")
    plt.scatter(centroids[-1, 0], centroids[-1, 1],
                color="red", label="next centroid")
    plt.title('Select % d th centroid' % (centroids.shape[0]))
    plt.legend()
    plt.xlim(-1.1, 1.1)
    plt.ylim(-0.1, 1.1)
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

    def __init__(self, data: pandas.DataFrame, sample: float, parameter_dict: Dict[str, Any] = None):
        """
        :param data:
        数据集 pandas.DataFrame 或 pandas.core.frame.DataFrame 类型
        :param sample:
        在聚类中心以 sample 比例抽取样本
        :param parameter_dict:
        其他的参数字典，可以考虑的有最大迭代次数
        """
        self.data = data
        self.sample = sample
        self.parameter_dict = parameter_dict
        ''''''
        self.tol = 1e-4
        ''''''
        self.max_iter = 100
        '''将 self.data 的数据域(只含特征值，不含标签)转换为 numpy 矩阵'''
        self.array_data = None
        '''k 值是有标签的类别数'''
        self.k = 0
        '''中心点，是相同标签点的中心点'''
        self.center_point = None
        '''属性列表，包含标签名'''
        self.attribute_list = list(self.data.columns)
        '''k 个类质心的位置坐标'''
        self.centroids = None
        '''样本所处的类以及到该类质心的距离'''
        self.cluster_class = None
        '''最终的数据'''
        self.final_data = dict()
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
        '''先将原有数据的标签列去除'''
        tmp_data = self.data.drop(columns=self.attribute_list[-1], inplace=False)
        '''遍历标签分组'''
        for label, value in label_groups.items():
            '''求得每个标签下的所有标签距离的均值'''
            if label != "None":
                self.center_point.append(
                    list(tmp_data.iloc[list(value)].mean()))
        '''inplace=False，删除标签所在列不改变原数据，返回一个执行删除操作后的新 dataframe，将其转换为 numpy 矩阵'''
        self.array_data: numpy.mat = numpy.mat(tmp_data)
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
        tmp_max_iter = self.max_iter
        while cluster_changed and tmp_max_iter > 0:
            '''聚类结果变化最大迭代次数减去一，布尔类型置为 False'''
            tmp_max_iter -= 1
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
                    dist_juli = distance(centroids[j, :], self.array_data[i, :])
                    '''如果当前距离小于当前最小距离'''
                    if dist_juli < min_dist:
                        '''当前距离为最小距离，最小距离对应的索引为 j(第j类)'''
                        min_dist = dist_juli
                        min_index = j
                '''当前聚类结果第 i 个样本的聚类结果发生变化，cluster_changed 置为 True，继续进行聚类算法'''
                if abs(cluster_class[i, 0] - min_index) > self.tol:
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

    def k_means(self):
        """
        KMeans 聚类算法
        """
        '''聚类中心由 KMeans 聚类算法给出'''
        t1 = time.time()
        centroids_ = rand_cent(self.array_data, self.k)
        self.centroids, self.cluster_class = self.k_means_basic(centroids_)
        print("time: " + str(time.time() - t1))

    def k_means_pluses(self):
        """
        KMeans++ 聚类算法
        https://zhuanlan.zhihu.com/p/149978127
        """
        '''聚类中心由 KMeans++ 聚类算法给出'''
        t1 = time.time()
        centroids_ = rand_cent_pluses(self.array_data, self.k)
        self.centroids, self.cluster_class = self.k_means_basic(centroids_)
        print("time: " + str(time.time() - t1))

    def k_means_m(self):
        """
        KMeans 改进初始聚类中心的聚类算法
        :return:
        """
        t1 = time.time()
        '''聚类中心由改进的 KMeans 聚类算法给出'''
        self.centroids, self.cluster_class = self.k_means_basic(self.center_point)
        print("time: " + str(time.time() - t1))

    def print_result(self, label_index: dict):
        """
        根据已经有的结果，对比计算得到的聚类结果，返回相应的信息
        :param label_index:
        :return:
        """
        i = 0
        res = 0
        '''将传入的字典转化为列表，列表每一行元素，第一个是标签序，第二个是索引长度，第三个是索引集合'''
        label_list = list(range(len(label_index)))
        '''矩阵维度'''
        m = numpy.shape(self.array_data)[0]
        '''转换'''
        for key, value in label_index.items():
            label_list[i] = (i, len(value), set(value))
            i += 1
        '''按列表元素的第二个元素升序排序'''
        label_list = sorted(label_list, key=lambda x: x[1], reverse=True)
        result_list = list(range(self.k))
        '''遍历所有标签'''
        for cent in range(self.k):
            '''获取所有该标签的索引'''
            index_list = list(numpy.nonzero(self.cluster_class[:, 0].A == cent)[0])
            result_list[cent] = (cent, len(index_list), set(index_list))
            '''输出每一个标签的平均SSE'''
            print("label " + str(cent) + ", SSE: " + str(
                sum(self.cluster_class[index_list][:, 1])[0, 0] / len(index_list)))
        '''输出所有标签的平均SSE'''
        print("label, SSE: " + str(
            sum(self.cluster_class[:, 1])[0, 0] / m))
        '''按列表元素的第二个元素升序排序'''
        result_list = sorted(result_list, key=lambda x: x[1], reverse=True)
        '''对两个列表的每个索引集合求交集'''
        for index in range(self.k):
            res += len((result_list[index][2]).intersection(label_list[index][2]))
        print("result: " + str(res / m))

    def show_result(self):
        """
        可视化结果，如果是(一定是)多维数据，可以按特征类型将数据划分为两类
        一类是所有离散特征的距离，一类是所有连续特征的距离
        :return:
        """

    def data_divide(self):
        """
        在 centroids 聚类中心，以 sample 比例抽取数据
        因为 self.cluster_class 记录了样本所处的类以及到该类质心的距离
        因此我们只需要按类别遍历
        先将同一类所在行索引和距离提取成为一个字典，键为行索引，值为距离
        之后按值升序排序，按比例选取前若干行索引
        得到 self.final_data
        :return:
        """
        '''遍历所有标签'''
        for cent in range(self.k):
            '''提取属于该类的所有行索引'''
            index_list = list(numpy.nonzero(self.cluster_class[:, 0].A == cent)[0])
            '''提取这些含行索引的距离'''
            index_value = [i for item in self.cluster_class[index_list][:, 1].tolist() for i in item]
            '''组合为字典'''
            index_dict = dict(zip(index_list, index_value))
            '''按字典的值排序'''
            l = sorted(index_dict.items(), key=lambda item: item[1], reverse=False)
            '''得到每个类需要抽取的样本数'''
            num = int(self.sample * len(index_list)) + 1
            '''收集前 num 个行索引'''
            self.final_data[cent] = [
                l[index][0] for index in range(num)
            ]
