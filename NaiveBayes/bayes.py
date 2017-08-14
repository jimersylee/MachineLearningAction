# -*- coding: utf-8 -*-
# @Time    : 17-8-14 上午11:28
# @Author  : Jimersy Lee
# @Site    : 
# @File    : bayes.py
# @Software: PyCharm
# @Desc    :

def loadDataSet():
    """
    定义一些数据,返回言论单词,以及言论性质
    :return:
    """
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字 0代表正常言论
    return postingList, classVec


def createVocabList(dataSet):
    """
    创建一个包含所有文档中出现的不重复单词的列表,为此使用了set数据类型
    :param dataSet:
    :return:
    """
    vocabSet = set([])  # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 创建两个集合的并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word:%s is not in my Vocabulary!" % word
    return returnVec


listOfPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOfPosts)
print myVocabList  # 输出的list没有重复的单词

# 检测每篇文章中哪些词条在字典中出现过

print setOfWords2Vec(myVocabList, listOfPosts[0])  # 第1篇文章哪些单词出现过
print setOfWords2Vec(myVocabList, listOfPosts[3])  # 第4篇文章哪些单词出现过
