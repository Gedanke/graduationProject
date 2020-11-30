# -*- coding: utf-8 -*-

import csv
import math
import numpy
import pandas
import operator
import treePlotter
from typing import List, Dict, Tuple
from collections import Counter
import copy

"""
该部分含有三种算法的不同实现
KNN 算法
决策书树算法
朴素贝叶斯算法

"""


class KNN(object):
    """
    KNN

    """

    def __init__(self, path: str, train_path: str, test_path: str, feature_dict: Dict[str, int], k: int):
        """
        :param path:
        数据集路径
        :param train_path:
        训练集数据路径
        :param test_path:
        测试集数据路径
        :param feature_dict:
        特征字典(dict)
        :param k:
        近邻数(int)
        """
        '''全体数据'''
        self.data = pandas.read_csv(path)
        '''训练集'''
        self.train_data = pandas.read_csv(train_path)
        '''测试集'''
        self.test_data = pandas.read_csv(test_path)
        '''特征字典'''
        self.feature_dict = feature_dict
        '''近邻数'''
        self.k = k
        '''特征列表'''
        self.attributes_list = list(self.test_data)
        '''标签名称'''
        self.label_name = self.attributes_list[-1]
        '''测试集标签'''
        self.test_label = list(self.test_data[self.label_name])
        '''特征列表,除去标签'''
        self.attributes_list.remove(self.label_name)
        '''样本标签'''
        self.label = sorted(list(set(self.train_data[self.label_name])))
        '''无序离散特征字典,不同标签下'''
        self.unordered_label_dict = {}
        '''无序离散特征字典,所有标签下'''
        self.unordered_all_dict = {}
        '''函数'''
        self.init_data()

    def init_data(self):
        """
        初始化
        :return:
        """
        self.test_data = self.test_data.drop(self.label_name, axis=1)
        tmp_dict = dict()
        for feature in self.feature_dict.keys():
            if self.feature_dict[feature] == 2:
                d = dict(
                    sorted(Counter(self.data[feature]).items(), key=operator.itemgetter(0)))
                self.unordered_all_dict[feature] = d
                tmp_dict[feature] = dict(zip(list(d), [0 for _ in range(len(d))]))
        '''标签数据集,按标签划分'''
        label_data = {}

        for label in self.label:
            label_data[label] = self.data[self.data[self.label_name] == label].drop(
                columns=[self.label_name])
            temp = {}
            tmp = copy.deepcopy(tmp_dict)
            for feature in self.feature_dict.keys():
                if self.feature_dict[feature] == 2:
                    d = dict(
                        sorted(Counter(label_data[label][feature]).items(), key=operator.itemgetter(0)))
                    d_ = tmp[feature]
                    for key in d.keys():
                        d_[key] = d[key]
                    temp[feature] = d_

            self.unordered_label_dict[label] = temp

    def distance(self, row):
        """
        距离度量,和Relief算法一致
        :return: 
        与样本 row 最近的k个样本(self.train_data)中出现最多的标签
        """
        '''训练数据集下的索引'''
        index_list = list(self.train_data.index)
        '''键是索引,值是样本 row 与 index索引样本上所有特征的距离度量和'''
        index_dict = {}
        for index in index_list:
            '''当前样本'''
            index_dict[index] = 0
            row_index = self.train_data.iloc[index]
            for feature in self.attributes_list:
                '''VDM'''
                if self.feature_dict[feature] == 2:
                    '''提前先各个标签下特征值在标签中的个数统计好'''
                    all_d = 0
                    c_j_1 = row[feature]
                    c_j_2 = row_index[feature]
                    for la in self.label:
                        all_d = abs(
                            self.unordered_label_dict[la][feature][c_j_1] / self.unordered_all_dict[feature][c_j_1] -
                            self.unordered_label_dict[la][feature][c_j_2] / self.unordered_all_dict[feature][c_j_2]
                        )
                    index_dict[index] += all_d

                else:
                    '''p=1 闵科夫斯基距离'''
                    '''该部分的数据已经归一化处理'''
                    index_dict[index] += abs(row[feature] - row_index[feature])
        sim = sorted(index_dict.items(), key=operator.itemgetter(1))[0:self.k]
        sim = dict(sim)
        sim_list = list(sim.keys())
        sim_dict = {}
        for sl in sim_list:
            sim_dict[sl] = self.train_data.iloc[sl][self.label_name]
        # sim_ = sorted(sim_dict.items(), key=operator.itemgetter(1))[-1][1]
        sim_ = Counter(sim_dict).most_common(1)[0][1]
        return sim_

    def get_result(self):
        """
        :return: 
        正确率
        """
        '''预测标签集'''
        predict_label = []
        right = 0
        '''遍历测试集'''
        for index in range(len(self.test_label)):
            '''选出其中一个样本'''
            row = self.test_data.iloc[index]
            predict_label.append(self.distance(row))
            if not isinstance(predict_label[index], str):
                if abs(predict_label[index] - self.test_label[index]) < 1e-6:
                    right += 1
            else:
                if predict_label[index] == self.test_label[index]:
                    right += 1
        return right / int(len(self.test_label))


def mean(feature) -> float:
    """
    :param feature:
    :return:
    特征均值
    """
    return sum(feature) / float(len(feature))


def sta_dev(feature) -> float:
    """
    :param feature:
    :return:
    特征标准差
    """
    mean_ = mean(feature)
    return math.sqrt(sum([pow(f - mean_, 2) for f in feature]) /
                     float(len(feature) - 1))


def summary(instances) -> List[Tuple[float, float]]:
    """
    :param instances:
    :return:
    summary_
    """
    summary_ = [(mean(feature), sta_dev(feature)) for feature in zip(*instances)]
    del summary_[-1]
    return summary_


def calculate_possibility(x, mean_, sta_dev_) -> float:
    """
    :param x:
    :param mean_:
    :param sta_dev_:
    :return:
    """
    exponent = math.exp(-(math.pow(x - mean_, 2) / (
            2 * math.pow(sta_dev_, 2))))
    return (1 / (math.sqrt(2 * math.pi) * sta_dev_)) * exponent


class NaiveBayes(object):
    """
    朴素贝叶斯(高斯分布)
    https://www.jianshu.com/p/d2745c85bbd4

    """

    def __init__(self, path_train: str, path_test: str):
        """
        :param path_train:
        训练数据集路径
        :param path_test:
        测试数据集路径
        """
        self.path_train = path_train
        self.path_test = path_test
        self.train_data = list()
        self.test_data = list()
        self.separated_data = dict()
        self.summary_data = dict()
        self.probability = dict()
        '''函数'''
        self.init_data()
        self.separate_summary()

    def init_data(self):
        """
        读取数据集
        :return:
        """
        '''读取训练集'''
        # train_lines = csv.reader(open(self.path_train, "r"))
        train_lines = csv.reader(open(self.path_train, "r", encoding="UTF-8-sig"))
        self.train_data = list(train_lines)
        self.train_data.remove(self.train_data[0])
        train_size = len(self.train_data)
        for index in range(train_size):
            self.train_data[index] = [float(x) for x in self.train_data[index]]
        '''读取测试集'''
        # test_lines = csv.reader(open(self.path_test, "r"))
        test_lines = csv.reader(open(self.path_test, "r", encoding="UTF-8-sig"))
        self.test_data = list(test_lines)
        self.test_data.remove(self.test_data[0])
        test_size = len(self.test_data)
        for index in range(test_size):
            self.test_data[index] = [float(x) for x in self.test_data[index]]

    def separate_summary(self):
        """
        按标签划分数据集
        :return:
        """
        for index in range(len(self.test_data)):
            sample = self.test_data[index]
            if sample[-1] not in self.separated_data:
                self.separated_data[sample[-1]] = []
            self.separated_data[sample[-1]].append(sample)
        for class_value, instances in self.separated_data.items():
            self.summary_data[class_value] = summary(instances)

    def calculate_possibility(self, sample):
        """
        :param sample:
        :return:
        """
        for classValue, classSummary in self.summary_data.items():
            self.probability[classValue] = 1
            for index in range(len(classSummary)):
                mean_, sta_dev_ = classSummary[index]
                x = sample[index]
                self.probability[classValue] *= calculate_possibility(x, mean_, sta_dev_)

    def predict(self, sample):
        """
        :return: 
        best_label
        """
        self.calculate_possibility(sample)
        best_label = None
        best_pro = -1
        for classValue, probability in self.probability.items():
            if best_label is None or probability > best_pro:
                best_pro = probability
                best_label = classValue
        return best_label

    def get_predictions(self):
        """
        :return: 
        predictions
        """
        predictions = []
        for index in range(len(self.test_data)):
            predictions.append(self.predict(self.test_data[index]))
        return predictions

    def get_result(self) -> float:
        """
        :return:
        正确率
        """
        correct = 0
        predictions = self.get_predictions()
        for index in range(len(self.test_data)):
            if self.test_data[index][-1] == predictions[index]:
                correct += 1
        return correct / float(len(self.test_data))


class NaiveBayesGauss(object):
    """
    朴素贝叶斯(高斯分布)
    https://www.cnblogs.com/HuZihu/p/10896677.html

    """

    def __init__(self, path_train: str, path_test: str):
        """
        :param path_train:
        训练数据集路径
        :param path_test:
        测试数据集路径
        """
        self.path_train = path_train
        self.path_test = path_test
        ''''''
        self.train_data = None
        self.test_data = None
        self.train_label = None
        self.test_label = None
        self.init_data()
        self.num_samples = None
        self.num_class = None
        self.label_list = list()
        self.prior_prob = list()
        self.sample_mean = list()
        self.sample_var = list()

    def init_data(self):
        """
        :return:
        """
        '''得到训练数据'''
        data = pandas.read_csv(self.path_train)
        lists = len(data.iloc[0])
        # data.iloc[:, 0:lists - 1] = data.iloc[:, 0:lists - 1].applymap(
        #     lambda x: numpy.NAN if x == 0 else x)
        # data = data.dropna(how="any")
        self.train_data = data.iloc[:, :-1]
        self.train_label = data.iloc[:, -1]
        '''得到测试数据'''
        data = pandas.read_csv(self.path_test)
        lists = len(data.iloc[0])
        # data.iloc[:, 0:lists - 1] = data.iloc[:, 0:lists - 1].applymap(
        #     lambda x: numpy.NAN if x == 0 else x)
        # data = data.dropna(how="any")
        self.test_data = data.iloc[:, :-1]
        self.test_label = data.iloc[:, -1]

    def separate(self):
        """
        :return: 
        data_class
        """
        self.num_samples = len(self.train_data)
        self.train_label = self.train_label.reshape(self.train_data.shape[0], 1)
        '''特征与标签合并'''
        data = numpy.hstack((self.train_data, self.train_label))
        data_class = dict()
        '''提取各类别数据,字典的键为类别名,值为对应的分类数据'''
        for index in range(len(data[:, -1])):
            if index in data[:, -1]:
                data_class[index] = data[data[:, -1] == index]
        self.train_label = numpy.asarray(self.train_label, numpy.float32)
        self.label_list = list(data_class.keys())
        self.num_class = len(data_class.keys())
        return data_class

    def gain_prior_prob(self, sample_label):
        """
        :param sample_label:
        :return:
        """
        return (len(sample_label) + 1) / (self.num_samples + self.num_class)

    def gain_sample_mean(self, sample):
        """
        :param sample:
        :return:
        """
        sample_mean = list()
        for index in range(sample.shape[1]):
            sample_mean.append(numpy.mean(sample[:, index]))
        return sample_mean

    def gain_sample_var(self, sample):
        """
        :param sample:
        :return:
        """
        sample_var = list()
        for index in range(sample.shape[1]):
            sample_var.append(numpy.var(sample[:, index]))
        return sample_var

    def gain_prob(self, sample, sample_mean, sample_var):
        """
        :param sample:
        :param sample_mean:
        :param sample_var:
        :return:
        """
        prob = list()
        for x, y, z in zip(sample, sample_mean, sample_var):
            prob.append((numpy.exp(-(x - y) ** 2 / (2 * z))) * (1 / numpy.sqrt(2 * numpy.pi * z)))
        return prob

    def train_model(self):
        """
        :return:
        """
        self.train_data = numpy.asarray(self.train_data, numpy.float32)
        self.train_label = numpy.asarray(self.train_label, numpy.float32)
        '''数据分类'''
        data_class = self.separate()
        '''计算各类别数据的目标先验概率,特征平均值和方差'''
        for data in data_class.values():
            sample = data[:, :-1]
            sample_label = data[:, -1]
            self.prior_prob.append(self.gain_prior_prob(sample_label))
            self.sample_mean.append(self.gain_sample_mean(sample))
            self.sample_var.append(self.gain_sample_var(sample))

    def predict(self, sample):
        """
        :return:
        """
        sample = numpy.asarray(sample, numpy.float32)
        poster_prob = list()
        idx = 0
        for x, y, z in zip(self.prior_prob, self.sample_mean, self.sample_var):
            gaussian = self.gain_prob(sample, y, z)
            poster_prob.append(numpy.log(x) + sum(numpy.log(gaussian)))
            idx = numpy.argmax(poster_prob)
        return self.label_list[idx]

    def get_result(self):
        """
        :return: 
        返回结果
        """
        self.train_model()
        acc = 0
        tp = 0
        fp = 0
        fn = 0
        for index in range(len(self.test_data)):
            '''对self.test_data 进行预测'''
            predict = self.predict(self.test_data.iloc[index, :])
            target = numpy.array(self.test_label)[index]
            if predict == 1 and target == 1:
                tp += 1
            if predict == 0 and target == 1:
                fp += 1
            if predict == target:
                acc += 1
            if predict == 1 and target == 0:
                fn += 1
        return acc / len(self.test_data), tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn)


def is_number(num):
    """
    :param num: 传入字符串
    :return: 
    判断字符串是否为一个数(整数,浮点数)
    """
    try:
        float(num)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(num)
        return True
    except (TypeError, ValueError):
        pass
    return False


class DecisionNode:
    """
    树的节点

    """

    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        """
        :param col:
        待检验的判断条件所对应的列索引值
        :param value:
        为了使结果为True,当前列必须匹配的值
        :param results:
        保存的是针对当前分支的结果,字典类型
        :param tb:
        DecisionNode,对应于结果为true时,树上相对于当前节点的子树上的节点
        :param fb:
        DecisionNode,对应于结果为false时,树上相对于当前节点的子树上的节点
        """
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


class DecisionTree(object):
    """
    决策树
    https://zhuanlan.zhihu.com/p/20794583

    """

    def __init__(self, path_train: str, path_test: str):
        """
        :param path_train:
        训练数据集路径
        :param path_test:
        测试数据集路径
        列表中的数据均为str类型
        使用 is_number 可以判断一个字符串是否为数值型
        """
        '''训练集'''
        self.train_data = list(csv.reader(open(path_train, "r", encoding="UTF-8-sig")))
        '''去除第一行(列名)'''
        self.train_data.pop(0)
        '''测试集'''
        self.test_data = list(csv.reader(open(path_test, "r", encoding="UTF-8-sig")))
        '''去除第一行(列名)'''
        self.test_data.pop(0)
        '''构成出的树'''
        self.tree = None
        '''测试集行数'''
        self.rows = len(self.train_data)
        '''样本特征个数'''
        self.lists = len(self.train_data[0])
        '''测试集标签集'''
        self.test_label = list()
        '''预测得到的标签集'''
        self.result_label = list()
        self.init_data()

    def init_data(self):
        """
        数值化为数值的字符串
        :return:
        """
        '''训练集'''
        for i in range(len(self.train_data)):
            for j in range(len(self.train_data[i])):
                '''数字'''
                if is_number(self.train_data[i][j]):
                    self.train_data[i][j] = float(self.train_data[i][j])
        '''测试集'''
        for i in range(len(self.test_data)):
            for j in range(len(self.test_data[i])):
                if is_number(self.test_data[i][j]):
                    self.test_data[i][j] = float(self.test_data[i][j])
        '''拆分测试集'''
        for line in self.test_data:
            self.test_label.append(line.pop(self.lists - 1))

    def unique_counts(self, data):
        """
        :param data:
        :return:
        """
        results = dict()
        for row in data:
            r = row[len(row) - 1]
            if r not in results:
                results[r] = 0
            results[r] += 1
        return results

    def entropy(self, data):
        """
        :param data:
        :return:
        """
        ent = 0.0
        results = self.unique_counts(data)
        for r in results.keys():
            p = float(results[r]) / len(data)
            ent = ent - p * math.log2(p)
        return ent

    def divide_set(self, data, column, value):
        """
        :param data:
        :param column:
        :param value:
        :return: 
        数据集被拆分成的两个集合
        """
        '''数值型(含浮点数和整数型),str类型'''
        if is_number(value):
            def split_function(row):
                return row[column] >= value
        else:
            def split_function(row):
                return row[column] == value
        '''将数据集拆分成两个集合,并返回'''
        set1 = [row for row in data if split_function(row)]
        set2 = [row for row in data if not split_function(row)]
        return set1, set2

    def build_tree(self, data, function):
        """
        :param data:
        :param function:
        :return:
        """
        if len(data) == 0:
            return DecisionNode()
        current_score = function(data)

        '''定义一些变量以记录最佳拆分条件'''
        best_gain = 0.0
        best_criteria = None
        best_sets = None

        column_count = len(data[0]) - 1
        for col in range(0, column_count):
            column_values = dict()
            for row in data:
                column_values[row[col]] = 1
            for value in column_values.keys():
                (set1, set2) = self.divide_set(data, col, value)

                '''信息增益'''
                p = float(len(set1)) / len(data)
                gain = current_score - p * function(set1) - (1 - p) * function(set2)
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)

        '''创建子分支'''
        if best_gain > 0:
            true_branch = self.build_tree(best_sets[0], function)
            false_branch = self.build_tree(best_sets[1], function)
            return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                                tb=true_branch, fb=false_branch)
        else:
            return DecisionNode(results=self.unique_counts(data))

    def print_tree_(self, tree, indent=""):
        """
        :param tree
        :param indent:
        :return:
        """
        '''判断是否是叶子节点'''
        if tree.results is not None:
            print(str(tree.results))
        else:
            '''打印判断条件'''
            print(str(tree.col) + ":" + str(tree.value) + "?")
            '''打印分支'''
            print(indent + "T->", end=" ")
            self.print_tree_(tree.tb, indent + " ")
            print(indent + "F->", end=" ")
            self.print_tree_(tree.fb, indent + " ")

    def print_tree(self):
        """
        可以直接使用上面那个函数
        个人使用 C++ 的习惯
        :return:
        """
        self.print_tree_(self.tree)

    def classify(self, sample, tree):
        """
        :param sample: 样本(不含标签)
        :param tree: 树
        :return:
        """
        if tree.results is not None:
            return tree.results
        else:
            node_value = sample[tree.col]
            if is_number(node_value):
                if node_value >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if node_value == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.classify(sample, branch)

    def gain_impurity(self, data):
        """
        :param data:
        :return:
        基尼不纯度
        随机放置的数据项出现于错误分类中的概率
        """
        total = len(data)
        counts = self.unique_counts(data)
        imp = 0
        for k in counts.keys():
            p1 = float(counts[k]) / total
            imp += p1 * (1 - p1)
        return imp

    def pruning_(self, tree, impurity):
        """
        :param tree:
        :param impurity:
        :return:
        剪枝
        """
        '''如果分支不是叶节点,则对其进行剪枝'''
        if tree.tb.results is None:
            self.pruning_(tree.tb, impurity)
        if tree.fb.results is None:
            self.pruning_(tree.fb, impurity)
        '''如果两个子分支都是叶节点,判断是否能够合并'''
        if tree.tb.results is not None and tree.fb.results is not None:
            '''构造合并后的数据集'''
            tb = list()
            fb = list()
            for v, c in tree.tb.results.items():
                tb += [[v]] * c
            for v, c in tree.fb.results.items():
                fb += [[v]] * c
            '''检查熵的减少量'''
            decrease = self.entropy(tb + fb) - (self.entropy(tb) + self.entropy(fb) / 2)
            if decrease < impurity:
                '''合并分支'''
                tree.tb = None
                tree.fb = None
                tree.results = self.unique_counts(tb + fb)

    def pruning(self, impurity):
        """
        :param impurity:
        :return:
        """
        self.pruning_(self.tree, impurity)

    def get_result(self, args):
        """
        :param args: 参数列表 第一个为函数名
        :return: 
        正确率
        """
        correct = 0
        self.tree = self.build_tree(self.train_data, args[0])
        '''剪枝'''
        if len(args) > 1:
            self.pruning(args[1])
        len_test_data = len(self.test_data)
        for index in range(len_test_data):
            sample_label = self.classify(self.test_data[index], self.tree)
            sample_label = list(sample_label)[0]
            self.result_label.append(sample_label)
            if sample_label == self.test_label[index]:
                correct += 1
        return correct / len_test_data


"""
https://www.cnblogs.com/wsine/p/5180315.html
https://segmentfault.com/a/1190000008563018
https://segmentfault.com/a/1190000012328603

"""


class ClassifyTree(object):
    def __init__(self, path_train: str, path_test: str):
        """
        :param path_train:
        训练数据集路径
        :param path_test:
        测试数据集路径
        """
        self.train_data = list(csv.reader(open(path_train, "r")))
        self.attributes = self.train_data.pop(0)
        self.attributes.pop(-1)
        self.test_data = list(csv.reader(open(path_test, "r")))
        self.test_data.pop(0)
        self.test_label = list()
        self.result_label = list()
        self.init_data()

    def init_data(self):
        """
        初始化数据
        :return:
        """
        '''训练集'''
        for i in range(len(self.train_data)):
            for j in range(len(self.train_data[i])):
                '''数字'''
                if is_number(self.train_data[i][j]):
                    self.train_data[i][j] = float(self.train_data[i][j])
        '''测试集'''
        for i in range(len(self.test_data)):
            self.test_label.append(self.test_data[i].pop(-1))
            for j in range(len(self.test_data[i])):
                if is_number(self.test_data[i][j]):
                    self.test_data[i][j] = float(self.test_data[i][j])

    def gain_entropy(self, data_set):
        """
        计算给定数据集的香农熵
        熵越大,数据集的混乱程度越大
        :param data_set: 数据集
        :return: 
        数据集的香农熵
        """
        num_entropy = len(data_set)
        label_counts = dict()
        for feat_vec in data_set:
            current_label = feat_vec[-1]
            if current_label not in label_counts.keys():
                label_counts[current_label] = 0
            label_counts[current_label] += 1
        entropy = 0.0
        for key in label_counts:
            prob = float(label_counts[key]) / num_entropy
            entropy -= prob * math.log(prob, 2)
        return entropy

    def split_data_set(self, data_set, axis, value):
        """
        按照给定特征划分数据集,去除选择维度中等于选择值的项
        :param data_set: 数据集
        :param axis: 选择维度
        :param value: 选择值
        :return: 
        划分数据集
        """
        ret_data_set = list()
        for feat_vec in data_set:
            if feat_vec[axis] == value:
                reduce_feat_vec = feat_vec[:axis]
                reduce_feat_vec.extend(feat_vec[axis + 1:])
                ret_data_set.append(reduce_feat_vec)
        return ret_data_set

    def gain_best_split_feature(self, data_set):
        """
        选择最好的数据集划分维度
        :param data_set: 数据集
        :return: 
        最好的划分维度
        """
        num_features = len(data_set[0]) - 1
        base_entropy = self.gain_entropy(data_set)
        best_info_ratio = 0.0
        best_feature = -1
        for i in range(num_features):
            feat_list = [example[i] for example in data_set]
            unique_val = set(feat_list)
            new_entropy = 0.0
            split_info = 0.0
            for value in unique_val:
                sub_data_set = self.split_data_set(data_set, i, value)
                prob = len(sub_data_set) / float(len(data_set))
                new_entropy += prob * self.gain_entropy(sub_data_set)
                split_info += -prob * math.log(prob, 2)
            info_gain = base_entropy - new_entropy

            if split_info == 0:
                continue
            info_gain_ratio = info_gain / split_info
            if info_gain_ratio > best_info_ratio:
                best_info_ratio = info_gain_ratio
                best_feature = i
        return best_feature

    def major_count(self, class_list):
        """
        数据集已经处理了所有属性,但是类标签依然不是唯一的
        采用多数判决的方法决定该子节点的分类
        :param class_list: 分类类别列表
        :return: 
        子节点的分类
        """
        class_count = dict()
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] = 1
        sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sort_class_count[0][0]

    def create_tree(self, data_set, attributes):
        """
        递归构建决策树
        :param data_set: 数据集
        :param attributes: 特征标签
        :return: 
        决策树
        """
        class_list = [example[-1] for example in data_set]
        if class_list.count(class_list[0]) == len(class_list):
            '''类别完全相同,停止划分'''
            return class_list[0]
        if len(data_set[0]) == 1:
            '''遍历完所有特征时返回出现次数最多的'''
            return self.major_count(class_list)

        best_feature = self.gain_best_split_feature(data_set)
        best_feature_attribute = attributes[best_feature]
        tree = {
            best_feature_attribute: {}
        }
        del (attributes[best_feature])
        '''得到列表包括节点所有的属性值'''
        feat_val = [example[best_feature] for example in data_set]
        unique_val = set(feat_val)
        for value in unique_val:
            sub_attributes = attributes[:]
            tree[best_feature_attribute][value] = self.create_tree(self.split_data_set(data_set, best_feature, value),
                                                                   sub_attributes)
        return tree

    def classify(self, tree, feat_attributes, test_vec):
        """
        跑决策树
        :param tree: 决策树
        :param feat_attributes: 分类标签
        :param test_vec: 测试数据
        :return: 
        决策结果
        """
        first_str = list(tree.keys())[0]
        second_dict = tree[first_str]
        feat_index = feat_attributes.index(first_str)
        class_label = None
        for key in second_dict.keys():
            if test_vec[feat_index] == key:
                if type(second_dict[key]).__name__ == "dict":
                    class_label = self.classify(second_dict[key], feat_attributes, test_vec)
                else:
                    class_label = second_dict[key]
        return class_label

    def classify_all(self, tree):
        """
        跑决策树
        :param tree: 决策树
        :return: 
        决策结果
        """
        correct = 0
        index = 0
        for test_vec in self.test_data:
            result = self.classify(tree, self.attributes, test_vec)
            self.result_label.append(result)
            if self.test_label[index] == self.result_label[index]:
                correct += 1
            index += 1
        return correct / len(self.test_label)

    def get_result(self):
        """
        得到结果并绘制树
        :return: 
        result
        """
        data_set = self.train_data
        labels_tmp = self.attributes[:]
        decision_tree = self.create_tree(data_set, labels_tmp)
        treePlotter.create_plot(decision_tree)
        result = self.classify_all(decision_tree)
        return result
