#-*- coding: utf-8 -*-

from numpy import *
from os import listdir
import operator

# 读取数据到矩阵
def img2vector(filename):
    # 创建向量
    returnVect = zeros((1,1024))
    
    # 打开数据文件，读取每行内容
    fr = open(filename)

    for i  in range(32):
        # 读取每一行
        lineStr = fr.readline()
        
        # 将每行前32字符转成int存入向量
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])

    return returnVect

# kNN算法实现    
def classify0(inX, dataSet, labels, k):
    # 获取样本数据数量
    dataSetSize = dataSet.shape[0]
    
    # 矩阵运算，计算测试数据与每个样本数据对应数据项的差值
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    
    # sqDistances 上一步骤结果平方和
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    
    # 取平方根，得到距离向量
    distances = sqDistances**0.5
    
    # 按照距离从低到高排序
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    
    # 依次取出最近的样本数据
    for i in range(k):
        # 记录该样本数据所属的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    
    # 对类别出现的频次进行排序，从高到低
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    
    # 返回出现频次最高的类别
    return sortedClassCount[0][0]

# 算法测试    
def handwritingClassTest():
    # 样本数据的类标签列表
    hwLabels = []
    
    # 样本数据文件列表
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    
    # 初始化样本数据矩阵（M*1024）
    trainingMat = zeros((m,1024))
    
    # 依次读取所有样本数据到数据矩阵
    for i in range(m):
        # 提取文件名中的数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        
        # 将样本数据存入矩阵
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    
    # 循环读取测试数据
    testFileList = listdir('digits/testDigits')
    
    # 初始化错误率
    errorCount = 0.0
    mTest = len(testFileList)
    
    # 循环测试每个测试数据文件
    for i in range(mTest):
        # 提取文件名中的数字
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        
        # 提取数据向量
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        
        # 对数据文件进行分类
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        
        # 打印KNN算法分类结果和真实的分类
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        
        # 判断KNN算法结果是否准确
        if (classifierResult != classNumStr): errorCount += 1.0
    
    # 打印错误率
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

# 执行算法测试
handwritingClassTest()