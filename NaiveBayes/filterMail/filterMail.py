# -*- coding: utf-8 -*-
# @Time    : 17-8-21 上午11:01
# @Author  : Jimersy Lee
# @Site    : 
# @File    : filterMail.py
# @Software: PyCharm
# @Desc    :使用朴素贝叶斯过滤垃圾邮件

import re
from numpy import *


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
    """
    词集转为向量,用于词集模型
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为1的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word:%s is not in my Vocabulary!" % word
    return returnVec


def bagOfWords2Vec(vocabList, inputSet):
    """
    词袋转为向量,用于词袋模型
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1  # 每遇到一个单词时,增加词向量中的对应值,而不像词集模型中,只是将对应的数值设为1
        else:
            print "the word:%s is not in my Vocabulary!" % word
    return returnVec


def trainNormalBayes0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix: 每篇文档中词是否出现的向量组成的矩阵 [[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]]
    0代表在所有单词中没有出现,1代表在所有单词中出现
    :param trainCategory:每篇文档列表类别标签所构成的向量 [0,1,0,1,0,1] 0代表非侮辱性 1代表侮辱性
    :return:
    """
    numTrainDocs = len(trainMatrix)  # 训练的文档数
    numWords = len(trainMatrix[0])  # 单词表中总的单词个数32个
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 3/6=0.5
    # p0Num = zeros(numWords)
    # p1Num = zeros(numWords)
    # 优化1:利用贝叶斯分类器进行分类时,要计算多个概率的乘积,如果其中一个概率值为0,那么最后的乘积也是0.为降低这种影响,可以将所有的词的出现数初始化为1,并将分母初始化为2
    p0Num = ones(numWords)  # 初始化numWords个元素的的数组,每个元素都是1
    # print p0Num
    p1Num = ones(numWords)
    p0Denom = 2.0  # 初始化类别0的计数值 分母初始化为2,消除概率为0的影响
    p1Denom = 2.0  # 初始化类别1的计数值 侮辱性言论类别
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 对数组中的每个元素都进行加操作
            p1Denom += sum(trainMatrix[i])  # 计数器=计数器+测试矩阵中的元素和
        else:
            p0Num += trainMatrix[i]  # 对数组中的每个元素都进行加操作
            p0Denom += sum(trainMatrix[i])  # 计数器=计数器+测试矩阵中的元素和
    # p1Vect = p1Num / p1Denom  # 对每个元素除以该类别中的总词数,利用numpy可以直接实现
    # p0Vect = p0Num / p0Denom
    # 优化2:另一个问题是下溢出,这是由于太多很小的数相乘造成的.一种解决办法就是对乘积取自然对数,代数中ln(a*b)=ln(a)+ln(b);
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNormalBayes(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    使用朴素贝叶斯算法进行分类
    :param vec2Classify: 要分类的向量
    :param p0Vec: 使用trainNormalBayes计算出来的属于0类的概率
    :param p1Vec: 使用trainNormalBayes计算出来的属于1类的概率
    :param pClass1: 属于class1的概率
    :return:
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def TestMail2Tokens():
    return mail2Tokens("email/ham/6.txt")


def TestText2Tokens():
    mySent = 'This book is the best book!'
    print mySent.split()
    #  ['This', 'book', 'is', 'the', 'best', 'book!']
    # 切分的结果不错,但是标点也被当成了词的一部分
    # 使用正则表达式来切分句子,其中分隔符是除了单词,数字外的任意字符串
    regEx = re.compile('\\W*')
    mySent = 'This book is the best book!'
    listOfTokens = regEx.split(mySent)
    formatTokens = []
    for tok in listOfTokens:
        if len(tok) > 0:  # 过滤空字符串
            formatTokens.append(tok.lower())  # 大写转换成小写
    return formatTokens


def mail2Tokens(filePath):
    """
    邮件转换为词条
    :param filePath:文件路径
    :return: 词条数组
    """
    emailText = open(filePath).read()
    return text2Tokens(emailText)


def text2Tokens(text):
    """
    文本内容转换为词条
    1.过滤空字符串
    2.大写转换为小写
    :param text: 文本内容
    :return: 词条数组
    """
    regEx = re.compile('\\W*')
    listOfTokens = regEx.split(text)
    formatTokens = []
    for tok in listOfTokens:
        if len(tok) > 0:  # 过滤空字符串
            formatTokens.append(tok.lower())  # 大写转换成小写
    return formatTokens


def testSpam():
    """
    测试垃圾邮件分类正确率
    :return:
    """
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = mail2Tokens("email/spam/%d.txt" % i)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = mail2Tokens("email/ham/%d.txt" % i)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNormalBayes0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNormalBayes(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount / len(testSet))

# 测试代码
# TestText2Tokens()
# print TestMail2Tokens()

# 测试训练结果
testSpam()
