# -*- coding: utf-8 -*-
# @Time    : 17-9-4 上午9:58
# @Author  : Jimersy Lee
# @Site    : 
# @File    : logRegres.py
# @Software: PyCharm
# @Desc    :


from numpy import *


def loadDataSet():
    """
    载入测试数据
    :return:
    """
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 为了计算方便,将X0设置为1.0
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    """
    Sigmoid函数,单位阶跃函数
    :param inX:
    :return:
    """
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    """
    梯度上升算法
    :param dataMatIn: 输入的数据矩阵,存放的100*3的矩阵
    :param classLabels: 输入的数据类别矩阵
    :return:返回训练好的迭代次数
    """
    # 转换为NumPy矩阵类型
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()  # 初始为1*100的行向量,为了便于矩阵运算,使用transpose转置为列向量100*1

    m, n = shape(dataMatrix)  # 得到矩阵大小
    alpha = 0.001  # 目标移动的步长
    maxCycles = 500  # 最大迭代次数
    weights = ones((n, 1))

    for k in range(maxCycles):
        # 矩阵相乘,下面两行,计算真实类别与预测类别的差值,接下来就是按照该差值的方向调整回归系数
        h = sigmoid(dataMatrix * weights)  # 代表的不是一次乘积计算,事实上该运算包含了300次的乘积,变量h不是一个数,而是一个列向量,100
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def stocGradAscent0(dataMatIn, classLabels):
    m, n = shape(dataMatIn)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatIn[i] * weights))  # h是向量
        error = classLabels[i] - h  # error是向量
        weights = weights + alpha * error * dataMatIn[i]
    return weights


def stocGradAscent1(dataMatIn, classLabels, numIter=150):
    """
    改进的随机梯度上升算法
    :param dataMatIn:
    :param classLabels:
    :param numIter:
    :return:
    """
    m, n = shape(dataMatIn)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # alpha每次迭代时需要调整
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取更新
            h = sigmoid(sum(dataMatIn[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatIn[randIndex]
            del (dataIndex[randIndex])
    return weights


def plotBestFit(weights):
    """
    画出数据集和Logistic回归最佳拟合直线的函数
    :param weights:系数
    :return:
    """
    import matplotlib.pyplot as plt
    weights = weights.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c="green")
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    """
    疝气病马死亡分类测试
    :return:
    """
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[i]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount / numTestVec))
    print "the error rate of this test is: %f" % errorRate
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))


def testCal():
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    print weights
    """
    得到一组回归系数,它确定了不同类别数据之间的分割线
    [[ 4.12414349]
     [ 0.48007329]
     [-0.6168482 ]]
    """


def testGradAscent():
    """
    测试梯度上升算法,画图
    :return:
    """
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    plotBestFit(weights)


def testStocGradAscent0():
    """
    测试随机梯度上升算法,画图
    :return:
    """
    dataArr, labelMat = loadDataSet()
    weights = stocGradAscent0(dataArr, labelMat)
    plotBestFit(weights)


def testStocGradAscent1():
    """
    测试随机梯度上升算法,画图
    :return:
    """
    dataArr, labelMat = loadDataSet()
    weights = stocGradAscent1(dataArr, labelMat)
    plotBestFit(weights)


# testCal()

#testGradAscent()

# testStocGradAscent0()
multiTest()