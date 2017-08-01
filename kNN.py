# coding:utf8
from numpy import *
import operator

def img2vector(filename):
    # 创建向量

    returnVector = zeros((1, 1024))

    # 打开数据文件,读取每行内容
    fr=open(filename)

    for i in range(32):
        # 读取每一行
        lineStr=fr.readline()
        # 将每行前32字符转成int存入向量
        for j in range(32):
            returnVector[0,32*i+j]=int(lineStr[j])

    return returnVector

def classify0(inX,dataSet,labels,k):

    # 获取样本数据数量
    dataSetSize=dataSet.shape[0]

    # 矩阵运算,计算测试数据与每个样本数据对应数据的差值
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    # sqDistance 上一步骤结果平方和
    sqDiffMat=diffMat**2
    sqDistance=sqDiffMat.sum(axis=1)

    # 取平方根,得到距离向量


