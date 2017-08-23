# -*- coding: utf-8 -*-
# @Time    : 17-8-14 上午11:28
# @Author  : Jimersy Lee
# @Site    : 
# @File    : bayes.py
# @Software: PyCharm
# @Desc    :

from numpy import *
import feedparser


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


def text2Tokens(text):
    """
    文本内容转换为词条
    1.过滤空字符串
    2.大写转换为小写
    :param text: 文本内容
    :return: 词条数组
    """
    import re
    regEx = re.compile('\\W*')
    listOfTokens = regEx.split(text)
    formatTokens = []
    for tok in listOfTokens:
        if len(tok) > 0:  # 过滤空字符串
            formatTokens.append(tok.lower())  # 大写转换成小写
    return formatTokens


def calcMostFreq(vocabList, fullText):
    """
    计算出现频率最高的前30个词汇
    :param vocabList: 总的词汇表
    :param fullText: 文章内容
    :param number:取前几位,默认30位
    :return:
    """
    import operator  # 导入操作包
    freqDict = {}  # 初始化词频字典
    for token in vocabList:
        freqDict[token] = fullText.count(token)  # 某个词作为key,出现的次数为value

    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)  # 对词频字典进行排序
    return sortedFreq[:30]  # 返回词频字典的前30位


def localWords(feed1, feed0):
    """
    使用两个RSS源作为参数
    :param feed1:
    :param feed0:
    :return:
    """
    docList = []  # (文章的单词组成的数组)作为成员的数组 ,[['hello','world'],['test','something']]
    classList = []  # 分类数组
    fullText = []  # 文章的单词组成的数组,每个单词作为数组成员 ['hello','world','test','something']
    minLen = min(len(feed1['entries']), len(feed0['entries']))  # 获取最小长度,防止数组越界
    for i in range(minLen):
        # feed1的内容进行处理
        wordList = text2Tokens(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # feed0的内容进行处理
        wordList = text2Tokens(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    #  去除出现次数最多的那些词
    for pairW in top30Words:  # pairW二维数组 0位置:单词 1位置:出现次数
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])  #

    trainingSet = range(2 * minLen)
    testSet = []
    # 对训练数据与测试数据进行留存交叉验证的处理
    for i in range(20):
        randomIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randomIndex])
        del (trainingSet[randomIndex])
    trainMat = []
    trainingClass = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainingClass.append(classList[docIndex])

    p0V, p1V, pClass1 = trainNormalBayes0(array(trainMat), array(trainingClass))
    errorCount = 0
    for docIndex in testSet:
        wordVect = bagOfWords2Vec(vocabList, docList[docIndex])
        classifyResult = classifyNormalBayes(wordVect, p0V, p1V, pClass1)
        actualResult = classList[docIndex]
        if classifyResult != actualResult:
            errorCount += 1

    print 'the error rate is: ', float(errorCount) / len(testSet)
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF============================================="
    for item in sortedSF:
        print item[0]
    print "NY============================================="
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    for item in sortedNY:
        print item[0]


# 测试代码
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craiglist.org/stp/index.rss')
vocabList, pSF, pNY = localWords(ny, sf)

getTopWords(ny, sf)
