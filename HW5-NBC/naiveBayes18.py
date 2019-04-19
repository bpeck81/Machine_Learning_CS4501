# !/usr/bin/python
#USING PYTHON3
#bjp9pq

import sys
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import glob
import math as math
import collections
###############################################################################

def transfer(fileDj, vocabulary):
    with open(fileDj, 'r') as f:
        for line in f:
            words = line.split(' ')
            for word in words:
                if word == 'loving' or word == 'loved' or word == 'loves':
                    word = 'love'
                if word in vocabulary:
                    vocabulary[word] += 1
                else:
                    vocabulary['UNK'] += 1
    return vocabulary


def loadData(Path):
    d ={'love':0, 'wonderful':0, 'best':0, 'great':0, 'superb':0,
        'still':0,'beautiful':0, 'bad':0, 'worst':0, 'stupid':0,
        'waste':0, 'boring':0, '?':0, '!':0, 'UNK':0}

    neg_train = glob.glob(Path + 'training_set/neg/*.txt')
    pos_train = glob.glob(Path + 'training_set/pos/*.txt')
    neg_test = glob.glob(Path + 'test_set/neg/*.txt')
    pos_test = glob.glob(Path + 'test_set/pos/*.txt')
    Xtrain = []
    Xtest = []
    ytrain = []
    ytest = []
   # xy = [np.ndarray() for i in range(4)]

    for file in neg_train:
        words = list(transfer(file, d.copy()).values())
        Xtrain.append(words)
        ytrain.append(-1)
    for file in pos_train:
        words = list(transfer(file, d.copy()).values())
        Xtrain.append(words)
        ytrain.append(1)
    for file in neg_test:
        words = list(transfer(file, d.copy()).values())
        Xtest.append(words)
        ytest.append(-1)
    for file in pos_test:
        words = list(transfer(file, d.copy()).values())
        Xtest.append(words)
        ytest.append(1)
    Xtrain = np.asarray(Xtrain)
    ytrain = np.asarray(ytrain)
    Xtest = np.asarray(Xtest)
    ytest = np.asarray(ytest)
    return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):
    alpha = 1
    class_count = int(len(ytrain)/2)
    Xtrainneg = Xtrain[0:class_count,]
    Ytrainneg = ytrain[0:class_count,]
    Xtrainpos = Xtrain[class_count:,]
    Ytrainpos = ytrain[class_count:,]

    mega_pos_freq = Xtrainpos.sum(axis=0)
    word_count_pos = Xtrainpos.sum(axis=1)
    word_count_pos = word_count_pos.sum(axis=0)
    mega_neg_freq = Xtrainneg.sum(axis=0)
    word_count_neg = Xtrainneg.sum(axis=1)
    word_count_neg = word_count_neg.sum(axis=0)
    #calc theta with smoothing
    thetaPos = (mega_pos_freq + alpha)/(word_count_pos+ alpha* Xtrain.shape[1])
    thetaNeg = (mega_neg_freq + alpha)/(word_count_neg+ alpha* Xtrain.shape[1])

    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg):
    #p(d|c) = 1/theta * w
    #p(c|d) = p(d|c)p(c)
    yPredict = []
    for i in range(len(Xtest)):
        pcgived = .5
        ncgived = .5
        for j in range(len(Xtest[i])):
            if Xtest[i,j] !=0:
                pcgived += math.log2((thetaPos[j]) * Xtest[i,j])
                ncgived += math.log2((thetaNeg[j]) * Xtest[i,j])
        pred = 1 if pcgived > ncgived else -1
        yPredict.append(pred)
    yPredict = np.asarray(yPredict)
    acc_arr = yPredict - ytest
    count = collections.Counter(acc_arr)
    Accuracy = count[0]/float(len(acc_arr))
    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    mult = MultinomialNB()
    mult.fit(Xtrain, ytrain)
    p = mult.predict(Xtest)
    Accuracy = mult.score(Xtest,ytest)

    return Accuracy


# def naiveBayesMulFeature_testDirectOne(path,thetaPos, thetaNeg):
#   return yPredict


def naiveBayesMulFeature_testDirect(path, thetaPos, thetaNeg):
    yPredict = []

    return yPredict, Accuracy


def naiveBayesBernFeature_train(Xtrain, ytrain):
    pos_count = [0 for i in range(len(Xtrain.T))]
    neg_count = [0 for i in range(len(Xtrain.T))]
    class_count = int(len(ytrain)/2)

    for i in range(class_count):
        for j in range(len(Xtrain[i])):
            if Xtrain[i,j] != 0:
                pos_count[j] += 1
    thetaPosTrue = [(pos_count[i]+1)/(class_count+2) for i in range(len(pos_count))]

    for i in range(class_count, 2*class_count):
        for j in range(len(Xtrain[i])):
            if Xtrain[i,j] != 0:
                neg_count[j] += 1
    thetaNegTrue = [(neg_count[i]+1)/(class_count+2) for i in range(len(neg_count))]

    return thetaPosTrue, thetaNegTrue


def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    for i in range(len(Xtest)):
        pcgived = .5
        ncgived = .5
        for j in range(len(Xtest[i])):
            if Xtest[i, j] != 0:
                pcgived += math.log2((thetaPos[j]) * Xtest[i, j])
                ncgived += math.log2((thetaNeg[j]) * Xtest[i, j])
        pred = 1 if pcgived > ncgived else -1
        yPredict.append(pred)
    yPredict = np.asarray(yPredict)
    acc_arr = yPredict - ytest
    count = collections.Counter(acc_arr)
    Accuracy = count[0] / float(len(acc_arr))
    return yPredict, Accuracy


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python naiveBayes.py dataSetPath testSetPath")
        sys.exit()

    print("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]

    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)

    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print("thetaPos =", thetaPos)
    print("thetaNeg =", thetaNeg)
    print("--------------------")

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print("MNBC classification accuracy =", Accuracy)

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print("Sklearn MultinomialNB accuracy =", Accuracy_sk)

    #   yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg)
    #   print "Directly MNBC tesing accuracy =", Accuracy
    print("--------------------")

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print("thetaPosTrue =", thetaPosTrue)
    print("thetaNegTrue =", thetaNegTrue)
    print("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print("BNBC classification accuracy =", Accuracy)
    print("--------------------")

