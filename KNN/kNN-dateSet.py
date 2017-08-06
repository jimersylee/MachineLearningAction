#coding:utf8
"""
kNN算法分析约会数据

"""
from numpy import *
def file2matrix(filename,colNum):
    """
    文件数据转为矩阵
    :param filename:文件名
    :param colNum: 文件数据的列数
    :return:
    """
    #打开文件
    fileHandle=open(filename)
    # 得到文件行数
    arrayAllLines=fileHandle.readlines()
    numberOfLines=len(arrayAllLines)
    # 创建返回的NumPy矩阵
    returnMatrix=zeros(numberOfLines,colNum)
    classLabelVector=[]
    index=0
    for line in arrayAllLines:
        line=line.strip()
        listFromLine=line.strip('\t')
        returnMatrix[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1

    return returnMatrix,classLabelVector
