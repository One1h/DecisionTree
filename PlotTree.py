import matplotlib.pyplot as plt

# 定义matplotlib的字体
plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback']
# boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细,也可写作 decisionNode={boxstyle:'sawtooth',fc:'0.8'}
decisionNode = dict(boxstyle="round", fc="0.8")
# 定义决策树的叶子结点的描述属性
leafNode = dict(boxstyle="circle", fc="0.8")
# 定义决策树的箭头属性
arrow_args = dict(arrowstyle="<-")


# nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="bottom", ha="center",
                            bbox=nodeType, arrowprops=arrow_args)


# 获取叶节点的数目
def getNumLeafs(myTree):
    # 定义叶子结点数目
    numLeaf = 0
    # 得到根据第一个特征分类的结果
    nodes = myTree.child_nodes
    # 遍历得到的子节点
    for node in nodes:
        # 如果node为一个决策树结点，非子节点
        if node.child_nodes:
            # 则递归的计算nodes中的叶子结点数，并加到numLeafs上
            numLeaf += getNumLeafs(node)
        else:
            numLeaf += 1
    # 返回求的叶子结点数目
    return numLeaf


# 获取树的层数
def getTreeDepth(myTree):
    # 定义树的深度
    maxDepth = 0
    # 得到第一个特征分类的结果
    nodes = myTree.child_nodes
    for node in nodes:
        # 如果node为一个决策树结点
        if node.child_nodes:
            thisDepth = 1 + getTreeDepth(node)
        # 如果node为一个决策树结点，非子节点
        else:
            # 则将当前树的深度设为1
            thisDepth = 1
        # 比较当前树的深度与最大数的深度
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    # 返回树的深度
    return maxDepth


# 绘制中间文本
def plotMidText(cntrPt, parentPt, txtString):
    # 求中间点的横坐标
    xMid = (parentPt[0] - cntrPt[0]) / 2.5 + cntrPt[0]
    # 求中间点的纵坐标
    yMid = (parentPt[1] - cntrPt[1]) / 2.5 + cntrPt[1]
    # 绘制树结点
    createPlot.ax1.text(xMid, yMid, txtString)


# 绘制决策树
def plotTree(myTree, parentPt, nodeTxt):
    # 定义并获得决策树的叶子结点数
    numLeafs = getNumLeafs(myTree)
    # 得到第一个特征
    firstStr = myTree.keyword
    # 计算坐标，x坐标为当前树的叶子结点数目除以整个树的叶子结点数再除以3，y为起点
    cntrPt = (plotTree.xOff + (1.0 + numLeafs) / len(myTree.child_nodes) / plotTree.totalW, plotTree.yOff)
    # 绘制决策树结点，也是当前树的根结点
    if parentPt == (0, 0):
        parentPt = cntrPt
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 根据第一个特征找到子节点
    nodes = myTree.child_nodes
    # 因为进入了下一层，所以y的坐标要变 ，图像坐标是从左上角为原点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    # 遍历字节带你
    for node in nodes:
        # 如果node为一棵子决策树，非叶子节点
        if node.child_nodes:
            # 递归的绘制决策树
            plotTree(node, cntrPt, node.parent)
        # node为叶子结点
        else:
            # 计算叶子结点的横坐标
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            # 绘制叶子结点
            plotNode(node.keyword, (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            # 特征值
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, node.parent)
    # 计算纵坐标
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# 主函数 绘图
def createPlot(inTree):
    # 定义一块画布
    fig = plt.figure(1, facecolor='white')
    # 清空画布
    fig.clf()
    # 定义横纵坐标轴，无内容
    axprops = dict(xticks=[], yticks=[])
    # 绘制图像，无边框，无坐标轴
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # plotTree.totalW保存的是树的宽
    plotTree.totalW = float(getNumLeafs(inTree))
    # plotTree.totalD保存的是树的高
    plotTree.totalD = float(getTreeDepth(inTree))
    # 决策树起始横坐标
    plotTree.xOff = -0.5 / plotTree.totalW
    # 决策树的起始纵坐标
    plotTree.yOff = 1.0
    # 绘制决策树
    plotTree(inTree, (0, 0), '')
    # 显示图像
    plt.savefig('tree.jpg')
