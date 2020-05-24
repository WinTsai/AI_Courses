'''
This is an script to implementation of KNN Classifier
Author: Win_Tsai
Date:   2020.05.17
'''
import sys
from numpy import *
from os import listdir
from collections import Counter

'''turn the image to vector 32x32->1x1024
**** param: image name
**** image size is 32x32
'''
def img2vector(filename):
    returnVect = zeros((1,1024))
    image = open(filename)
    for i in range(32):
        lineStr = image.readline()
        for j in range(32):
            returnVect[0,32*i + j] = int(lineStr[j])
    return returnVect


''' using KNN method to classify image vector
*** inX: the image vector
*** dataSet: the input training samples
*** labels: the label of every training image vector 
*** K: the number of nearest neighbours to choose 
'''
def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # tile is filled the input image size same with trainingMat
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    # get squared diffMat
    sqDiffMat = diffMat ** 2
    # sum each row of Matrix, to get the squared distance between input vector and training samples.
    sqDistances = sqDiffMat.sum(axis=1)
    # sqrt, each row stands for the dist between input with different trian sample
    distances = sqDistances ** 0.5
    # arrange distance from little and return the index location
    sortedDistIndicies = distances.argsort()
    # to choose the minum-dist k points
    classCount = {}
    for i in range (k):
        # to find sample class
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel)
    # resort and return the class which appear most
    maxClassCount = max(classCount,key=classCount.get)
    return maxClassCount


def handWritingClassTest():
    # import data
    hwLabels = []
    trainingFileList = listdir('D:\\uidq1860\\Work-CJW\\LearningFiles\\GitHub\\DataSets\\Handwritings\\trainingDigits\\')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))  # m samples and 32x32 demensions
    # hwLabels record 0-9's location, traingMat record binary image pixels
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # trun 32x32 mat to 1x1024 mat
        trainingMat[i,:] = img2vector('D:\\uidq1860\\Work-CJW\\LearningFiles\\GitHub\\DataSets\\Handwritings\\trainingDigits\\%s' %fileNameStr)

    # import testing data
    testFileList = listdir('D:\\uidq1860\\Work-CJW\\LearningFiles\\GitHub\\DataSets\\Handwritings\\testDigits\\')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('D:\\uidq1860\\Work-CJW\\LearningFiles\\GitHub\\DataSets\\Handwritings\\testDigits\\%s' % fileNameStr)
        # k=1, only to find the nearest label of the input vector
        classifierResult = classify(vectorUnderTest, trainingMat,hwLabels,1)
        print('\n the classifier came back with: %d, the real answer is %d' %(classifierResult,classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print('\n the total number of wrror is %d' % errorCount)
    print('\n the total error rate is : %f' %(errorCount/float(mTest)))


handWritingClassTest()
