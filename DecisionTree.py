import math
from copy import copy
from graphviz import Digraph
from typing import List
import PlotTree as pt


# 建立数据集
def createDataSet():
    dataSet = [
        # 1
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 2
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 3
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 4
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 5
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 6
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        # 7
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        # 8
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
        # 9
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        # 10
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        # 11
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        # 12
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        # 13
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 14
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 15
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        # 16
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        # 17
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ]

    # 特征值列表
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']

    # 特征对应的所有可能的情况
    labels_full = {}

    for i in range(len(labels)):
        labelList = [example[i] for example in dataSet]
        uniqueLabel = set(labelList)
        labels_full[labels[i]] = uniqueLabel

    return dataSet, labels, labels_full


# 多叉树
class BTreeNode(object):
    def __init__(self, parent=None, keyword=None, child_nodes=[]):
        '''parent：上一层划分属性的具体属性值，如：”浅白“
        keyeyword：此节点的划分属性或label，如：“颜色”
        child_nodes:根据此节点属性的不同属性值划分的子节点集'''
        self.parent = parent
        self.keyword = keyword
        self.child_nodes = child_nodes

    def getkeyword(self):
        return self.keyword

    def addchild(self, node):
        self.child_nodes.append(node)

    def setkeyword(self, keyword):
        self.keyword = keyword

    def setparent(self, parent):
        self.parent = parent

    def shownode(self):
        print("parent:{}\nkeyword:{}\nchild_nodes: ".format(self.parent, self.keyword))
        for node in self.child_nodes:
            print(node.parent, node.keyword)
        print()


# 计算信息熵
def Entropy(pk: float) -> float:
    if pk == 0.0: return 0.0
    return -1 * pk * math.log(pk, 2)


# 计算信息增益
def Gain(D: List[int], Ent: float) -> float:
    G = Ent
    for Dv in D:
        G -= abs(Dv / sum(D)) * Entropy(Dv / sum(D))
    return G


# 获取最佳划分属性
def BestAttribute(dataSet_, labels_, labels_full_):
    # 根节点信息熵计算
    temp = []
    D_t = [0, 0]
    for i, data in enumerate(dataSet_):
        temp.append(i + 1)
        if data[-1] == '好瓜':
            D_t[0] += 1
        if data[-1] == '坏瓜':
            D_t[1] += 1
    Ent = Entropy(D_t[0] / len(temp)) + Entropy(D_t[1] / len(temp))

    # 初始化样本集和信息熵列表
    Gains = []
    for ind, label in enumerate(labels_):
        l = len(labels_full_[label])
        G = Ent
        label_t = list(labels_full_[label])
        D = []
        Ents = []
        for i in range(l):
            D.append([])
            Ents.append(0)

        # 按属性划分Dv
        for i, data in enumerate(dataSet_):
            attribute_ind = label_t.index(data[ind])
            D[attribute_ind].append(i + 1)

        # 计算Dv中各类别数量
        Dv = []
        for i in D:
            temp = [0, 0]
            for j in i:
                if dataSet_[j - 1][-1] == '好瓜':
                    temp[0] += 1
                if dataSet_[j - 1][-1] == '坏瓜':
                    temp[1] += 1
            Dv.append(temp)

        # 计算信息熵
        for i, data in enumerate(Dv):
            good, bad = data
            total = good + bad
            if total != 0:
                Ents[i] = Entropy(good / total) + Entropy(bad / total)

        # 计算信息增益
        for i, data in enumerate(Ents):
            G -= (Dv[i][0] + Dv[i][1]) / len(dataSet_) * data
        Gains.append(G)

    # 寻找最大信息熵的属性
    label_num = 0
    for i, g in enumerate(Gains):
        if g > Gains[label_num]:
            label_num = i

    return labels_[label_num], Gains[label_num]


# 若全为同一类别，返回此类叶结点
def SameClass(dataset_):
    # 若全为同一类别，返回此类叶结点
    label = ''
    same_class = True
    for i, data in enumerate(dataset_):
        if i == 0:
            continue
        if data[-1] != dataset_[i - 1][-1]:
            same_class = False
            break
    if same_class:
        label = dataset_[0][-1]

    return same_class, label


# 属性为空 或 样本在属性上取值相同
def NoneOrSameattr(dataset_, labels_):
    if labels_ != []:
        for i in range(len(dataset_)-2):
            for j in range(i, len(dataset_)-1):
                if dataset_[i][:-1] == dataset_[j][:-1]:
                    return False

    return True


# 返回最多类别
def MostClass(dataset_):
    good, bad = 0, 0
    for data in dataset_:
        if data[-1] == '好瓜':
            good += 1
        if data[-1] == '坏瓜':
            bad += 1
    label = '好瓜' if good >= bad else '坏瓜'

    return label


# 对属性划分后不同子集继续生成分支结点
def GetSubNode(dataset_, labels_, labels_full_, best_attr):
    root = BTreeNode(keyword=best_attr)
    subnodes = []
    ind = labels_.index(best_attr)
    # 根据划分属性的不同属性值，对不同属性值的子集进行子树生成
    for attr in labels_full_[best_attr]:
        subtree = BTreeNode()
        subdataset = []
        for i, data in enumerate(dataset_):
            if data[ind] == attr:
                temp = copy(data)
                temp.pop(ind)
                subdataset.append(temp)

        # 该属性值子集为空，设为样本最多的类别
        if not subdataset:
            label = MostClass(dataset_)
            subtree.setkeyword(label)

        # 该属性值子集不为空，继续进行子决策树生成
        else:
            sublabels_full = copy(labels_full_)
            if best_attr in sublabels_full:
                sublabels_full.pop(best_attr)

            sublabels = copy(labels_)
            if best_attr in sublabels:
                sublabels.remove(best_attr)

            subtree = TreeGenerate(subdataset, sublabels, sublabels_full)

        subtree.setparent(attr)
        subnodes.append(subtree)

    return subnodes


# 生成决策树
def TreeGenerate(dataset_, labels_, labels_full_):
    root = BTreeNode()
    # 若全为同一类别，返回此类叶结点
    flag, label = SameClass(dataset_)
    if flag:
        root.setkeyword(label)
        return root

    # 属性为空 或 样本在属性上取值相同，返回最多类别
    if NoneOrSameattr(dataset_, labels_):
        label = MostClass(dataset_)
        root.setkeyword(label)
        return root

    # 选择最优划分属性
    best_attr, gain = BestAttribute(dataset_, labels_, labels_full_)
    root.setkeyword(best_attr)

    # 对属性划分后不同子集继续生成分支结点
    root.child_nodes = GetSubNode(dataset_, labels_, labels_full_, best_attr)
    return root


# 决策树预测
def test(data, dataset, label, labels_full, tree):
    res = ''
    # 遍历决策树，直到得到label
    while res not in ['坏瓜', '好瓜']:
        # 获取划分属性
        attr_divide = tree.keyword
        ind = label.index(attr_divide)

        for node in tree.child_nodes:
            #根据属性值进行划分
            if node.parent == data[ind]:
                tree = node
                res = node.keyword
                break

    return res



if __name__ == '__main__':
    dataSet, labels, labels_full = createDataSet()
    tree = TreeGenerate(dataSet, labels, labels_full)
    pt.createPlot(tree)
    data = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜']
    print(test(data, dataSet, labels, labels_full, tree))
