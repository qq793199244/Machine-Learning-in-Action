from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


'''
对用k-近邻算法将每组数据划分到某个类中，其伪代码如下：
对未知类别属性的数据集中的每个点依次执行以下操作：
（1）计算已知类别数据集中的点与当前点之间的距离
（2）按照距离递增次序排序
（3）选取与当前点距离最小的k个点
（4）确定前k个点所在类别的出现频率
（5）返回前k个点出现频率最高的类别作为当前点的预测分类
'''
'''
4个输入参数：
（1）inX：用于分类的输入向量    （2）dataSet：输入的训练样本集
（3）labels：标签向量  （4）k：用于选择最近邻居的数目
'''
def classify0(inX, dataSet, labels, k):
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方
    sqDiffMat =diffMat ** 2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加，
    # sum默认参数为axis=0为普通相加，axis=1为一行的行向量相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方，计算出距离，欧氏距离
    distances = sqDistances ** 0.5
    # argsort返回distances中元素从小到大排序后的索引值，数组索引0,1,2,3）
    sortedDistIndicies = distances.argsort()
    # 定一个记录类别次数的字典
    classCount={}
    # 选择距离最小的k个点
    for i in range(k):
        # 取出前k个元素的类别，根据排序结果的索引值返回靠近的前k个标签
        voteIlabel = labels[sortedDistIndicies[i]]
        # 各个标签出现频率
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序；key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]


'''
将待处理的格式改变为分类器可以接受的格式
第一个数据集会报错，改用datingTest2.txt
'''
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m],3)
        print("the classifier came back with %d, the real answer  is: %d" % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is %f" % (errorCount/float(numTestVecs)))


# 约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    ainArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((ainArr - minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult - 1])
