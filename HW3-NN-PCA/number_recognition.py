#!/usr/bin/env python
#bjp9pq#
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import sys
from sklearn.decomposition import RandomizedPCA
import numpy as np
from sklearn import svm

def split_data(train, test):
    train_data = np.genfromtxt(train)
    test_data =  np.genfromtxt(test)
    y_train = train_data[:,0]
    x_train = train_data[:,1:]
    y_test = test_data[:,0]
    x_test = test_data[:,1:]
    return x_train, y_train, x_test, y_test

def format_y(y):
    s = ""
    for row in y:
        s += str(int(row)) + '\n'
    return s
def neural_net(train, test):
    y = []
    x_train, y_train, x_test, y_test = split_data(train,test)
    mlp = MLPClassifier()
    mlp.fit(x_train, y_train)
    y  = mlp.predict(x_test)
    #print(mlp.score(x_test,y_test))
    return format_y(y)

def knn(train, test):
    y = []
    x_train, y_train, x_test, y_test = split_data(train,test)
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y = knn.predict(x_test)
    #print(knn.score(x_test,y_test))
    return format_y(y)

#required
def pca_LG(train, test):
    y = []
    x_train, y_train, x_test, y_test = split_data(train,test)
    pca = RandomizedPCA(n_components=500)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y = lr.predict(x_test)
    #print(lr.score(x_train,y_train))
    return format_y(y)

#Required
def LogistRegres(train, test):
    x_train, y_train, x_test, y_test = split_data(train,test)
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y  = lr.predict(x_test)
    #print(lr.score(x_test,y_test))
    return format_y(y)

def pca_knn(train, test):
    y = []
    x_train, y_train, x_test, y_test = split_data(train,test)
    pca = RandomizedPCA(n_components=2)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y = knn.predict(x_test)
    #print(knn.score(x_test,y_test))
    return format_y(y)

def SVM(train,test):
     y = []
     x_train, y_train, x_test, y_test = split_data(train, test)
     s = svm.SVC()
     s.fit(x_train, y_train)
     y = s.predict(x_test)
     #print(s.score(x_test, y_test))
     return format_y(y)

if __name__ == '__main__':
    model = "SVM"
    train = sys.argv[1]
    test = sys.argv[2]

    if model == "knn":
        print(knn(train, test))
    elif model == "SVM":
        print(SVM(train,test))
    elif model == "net":
        print(neural_net(train, test))
    elif model == "pcaknn":
        print(pca_knn(train, test))
    elif model == "pcaLG":
        print(pca_LG(train, test))
    elif model == "LG":
        print(LogistRegres(train, test))
    else:
        print("Invalid method selected!")
