# -*- coding: utf-8 -*-

from mnist.loader import MNIST
from sklearn.cluster import KMeans
import pandas
import numpy

# data = MNIST('datasets')
# X_train, y_train = data.load_training()
#
# # do the clustering
# k_means = KMeans(n_clusters=len(numpy.unique(y_train)))
# k_means.fit(X_train)
# labels = k_means.labels_
#
# predict = k_means.predict(data)
# data['cluster'] = predict
# pandas.tools.plotting.parallel_coordinates(data, 'cluster')
from functools import cmp_to_key

d = [(0, 6, {8, 10, 11, 13, 15, 16}), (1, 11, {0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 14})]

print(d)


def func(x, y):
    if x[1] < y[1]:
        return -1
    elif x[1] == y[1]:
        return 0
    else:
        return 1


sorted(d, key=lambda x: x[1])
l = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 14]
print(len(d[1][2]))
# 普通k-means
import numpy as np
import matplotlib.pyplot as plt
def loadDataSet(fileName):
    '''
    :param fileName: 文件名字
    :return: 矩阵
    '''
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))
# 随机初始化质心
def randCent(dataSet, k):
    '''
    构建一个包含k个随机质心的集合（k行n列，n表示数据的维度/特征个数），
    只需要保证质心在数据边界里面就可以了
    :param dataSet: 输入数据集
    :param k: 质心个数
    :return:
    '''
    # 得到数据样本的维度
    n = np.shape(dataSet)[1]
    # 初始化为一个(k,n)的全零矩阵
    centroids = np.mat(np.zeros((k, n)))
    # 遍历数据集的每一个维度
    for j in range(n):
        # 得到该列数据的最小值,最大值
        minJ = np.min(dataSet[:, j])
        maxJ = np.max(dataSet[:, j])
        # 得到该列数据的范围(最大值-最小值)
        rangeJ = float(maxJ - minJ)
        # k个质心向量的第j维数据值随机为位于(最小值，最大值)内的某一值
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)  # random.rand(行，列)产生这个形状的矩阵，且每个元素in [0,1)
    # 返回初始化得到的k个质心向量
    return centroids
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    '''
    :param dataSet: 输入的数据集
    :param k: 聚类的个数，可调
    :param distMeas: 计算距离的方法，可调
    :param createCent: 初始化质心的位置的方法，可调
    :return: k个类质心的位置坐标，样本所处的类&到该类质心的距离
    '''
    # 获取数据集样本数
    m = np.shape(dataSet)[0]
    # 初始化一个（m,2）全零矩阵，用来记录没一个样本所属类，距离类中心的距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 创建初始的k个质心向量
    centroids = createCent(dataSet, k)
    # 聚类结果是否发生变化的布尔类型
    clusterChanged = True
    # 终止条件：所有数据点聚类结果不发生变化
    while clusterChanged:
        # 聚类结果变化布尔类型置为False
        clusterChanged = False
        # 遍历数据集每一个样本向量
        for i in range(m):
            # 初始化最小距离为正无穷，最小距离对应的索引为-1
            minDist = float('inf')
            minIndex = -1
            # 循环k个类的质心
            for j in range(k):
                # 计算数据点到质心的欧氏距离
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                # 如果距离小于当前最小距离
                if distJI < minDist:
                    # 当前距离为最小距离，最小距离对应索引应为j(第j个类)
                    minDist = distJI
                    minIndex = j
            # 当前聚类结果中第i个样本的聚类结果发生变化：布尔值置为True，继续聚类算法
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            # 更新当前变化样本的聚类结果和平方误差
            clusterAssment[i, :] = minIndex, minDist**2
        # 打印k-means聚类的质心
        # print(centroids)
        # 遍历每一个质心
        for cent in range(k):
            # 将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算这些数据的均值(axis=0:求列均值)，作为该类质心向量
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    # 返回k个聚类，聚类结果及误差
    return centroids, clusterAssment
# k-means++初始化质心
def initialize(dataSet, k):
    '''
    K-means++初始化质心
    :param data: 数据集
    :param k: cluster的个数
    :return:
    '''
    # 得到数据样本的维度
    n = np.shape(dataSet)[1]
    # 初始化为一个(k,n)的全零矩阵
    centroids = np.mat(np.zeros((k, n)))
    # step1: 随机选择样本点之中的一个点
    centroids[0, :] = dataSet[np.random.randint(dataSet.shape[0]), :]  # np.random.randint()
    # plotCent(data, np.array(centroids))
    # 迭代
    for c_id in range(k - 1):
        dist = []
        for i in range(dataSet.shape[0]):  # 遍历所有点
            point = dataSet[i, :]
            d = sys.maxsize
            for j in range(centroids.shape[0]):  # 扫描所有质心，选出该样本点与最近的类中心的距离
                temp_dist = distEclud(point, centroids[j, :])
                d = min(d, temp_dist)
            dist.append(d)
        dist = np.array(dist)
        next_centroid = dataSet[np.argmax(dist), :]  # 返回的是dist里面最大值的下标，对应的是上面循环中的i
        centroids[c_id+1, :] = next_centroid  # 选出了下一次的聚类中心，开始k+1循环
        dist = []
        #plotCent(data, np.array(centroids))
    return centroids
# k-means++聚类
def kMeansPlus(dataSet, k, distMeas=distEclud, createCent=initialize):
    '''
    :param dataSet: 输入的数据集
    :param k: 聚类的个数，可调
    :param distMeas: 计算距离的方法，可调
    :param createCent: 初始化质心的位置的方法，可调
    :return: k个类质心的位置坐标，样本所处的类&到该类质心的距离
    '''
    # 获取数据集样本数
    m = np.shape(dataSet)[0]
    # 初始化一个（m,2）全零矩阵，用来记录没一个样本所属类，距离类中心的距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 创建初始的k个质心向量
    centroids = createCent(dataSet, k)
    # 聚类结果是否发生变化的布尔类型
    clusterChanged = True
    # 终止条件：所有数据点聚类结果不发生变化
    while clusterChanged:
        # 聚类结果变化布尔类型置为False
        clusterChanged = False
        # 遍历数据集每一个样本向量
        for i in range(m):
            # 初始化最小距离为正无穷，最小距离对应的索引为-1
            minDist = float('inf')
            minIndex = -1
            # 循环k个类的质心
            for j in range(k):
                # 计算数据点到质心的欧氏距离
                distJI = distMeas(centroids[j], dataSet[i])
                # 如果距离小于当前最小距离
                if distJI < minDist:
                    # 当前距离为最小距离，最小距离对应索引应为j(第j个类)
                    minDist = distJI
                    minIndex = j
            # 当前聚类结果中第i个样本的聚类结果发生变化：布尔值置为True，继续聚类算法
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            # 更新当前变化样本的聚类结果和平方误差
            clusterAssment[i, :] = minIndex, minDist**2
        # 打印k-means聚类的质心
        # print(centroids)
        # 遍历每一个质心
        for cent in range(k):
            # 将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算这些数据的均值(axis=0:求列均值)，作为该类质心向量
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    # 返回k个聚类，聚类结果及误差
    return centroids, clusterAssment
# 计算误差对比
dataMat = np.mat(loadDataSet('testSet2.txt'))
# 简单k-means
sumSSE = 0
for i in range(40):
    myCentroids2, clustAssing2 = kMeans(dataMat, 4)
    sumSSE += sum(clustAssing2[:, 1])
avgSSE = sumSSE/40
print('Avg(40) SSE of simple k-means: %f' % avgSSE)
# k-means++
sumSSE2 = 0
for i in range(40):
    myCentroids3, clustAssing3 = kMeansPlus(dataMat, 4)
    sumSSE2 += sum(clustAssing3[:, 1])
avgSSE2 = sumSSE2/40
print('Avg(40) SSE of simple k-means++: %f' % avgSSE2)