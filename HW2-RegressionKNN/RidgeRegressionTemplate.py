# Machine Learning HW2-Ridge

__author__ = 'bjp9pq'

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import operator
from matplotlib.ticker import LinearLocator, FormatStrFormatter# more imports


def loadDataSet(filename):
    data = np.genfromtxt(filename, delimiter=' ')
    np.random.seed(37)
    np.random.shuffle(data)
    x = data[:,0:3]
    y = data[:,3]
    return x, y

def draw3DPlot(xVal, beta):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    density = xVal.shape[0]
    X = np.linspace(-3,3,density).reshape((density,1))
    Y = np.linspace(-7,7,density).reshape((density,1))
    X, Y = np.meshgrid(X, Y)
    Z = beta[0]+ X*beta[1] + Y*beta[2]

    #Z = beta[0]+ (X.T * xVal[:,1]).T *beta[1] + (Y.T * xVal[:,2]).T *beta[2]
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=1, antialiased=False)
    ax.set_xlabel("Beta 1")
    ax.set_ylabel("Beta 2")
    ax.set_zlabel("Y")
    return

def ridgeRegress(xVal, yVal, lambdaV, showFigure=True):
    beta = np.dot(np.dot(np.linalg.inv(np.dot(xVal.T, xVal) + lambdaV * np.eye(xVal.shape[1])), xVal.T),yVal)
    if showFigure:
        draw3DPlot(xVal,beta)
    return beta



def calculateError(predicted, yVal):
    sum = 0
    for i in range(len(predicted)):
        sum += (predicted[i] - yVal[i]) ** 2
    mse = float(sum)/ len(predicted)
    return mse

def splitData(xVal, yVal, k, kfold):
    yVal = yVal.reshape(yVal.shape[0],1)
    data = np.hstack((xVal, yVal))
    remainder = data.shape[0] %kfold
    fold_size = data.shape[0] // kfold
    start_row = k * fold_size
    additive  = 1 if k+1 <= remainder else 0
    start_row += k if k+1 <= remainder else remainder
    testing = data[start_row:start_row+fold_size+additive]
    training = data.copy()
    training = np.delete(training, np.s_[start_row:start_row + fold_size + additive], axis=0)
    return training, testing

def JBVLPlot(lambda_errors):
    jb = []
    for i in range(len(lambda_errors)):
        jb.append(lambda_errors[i][1])
    jb = np.asarray(jb)
    lam = np.arange(0,1,.02)
    fig = plt.figure()
    ax = fig.gca()
  #  ax.title ='Path of Finding Best Lambda'
    ax.set_xlabel('Lambda')
    ax.set_ylabel('J(B)')
    ax.scatter(lam, jb)
    return


def cv(xVal, yVal):
    kfold = 10
    lambda_errors = {}
    for lambdaV in np.arange(0,1,.02):
        error_sum = 0
        for k in range(kfold):
            training, testing = splitData(xVal, yVal, k, kfold)
            training_x = training[:,0:3]
            training_y = training[:,3]
            beta = np.dot(np.dot(np.linalg.inv(np.dot(training_x.T, training_x) + lambdaV * np.eye(training_x.shape[1])), training_x.T), training_y)
            y_pred = beta[0] + testing[:,1]* beta[1] + testing[:,2]*beta[2]
            error_sum += calculateError(y_pred, testing[:,3])
        lambda_errors[lambdaV] = error_sum/float(kfold)
    sorted_votes = sorted(lambda_errors.iteritems(), key=operator.itemgetter(1), reverse=False)
    JBVLPlot(sorted_votes)
    lambdaBest  = sorted_votes[0][0]
    return lambdaBest



def standRegress(xVal, yVal):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(xVal.T, xVal)), xVal.T), yVal)
    linex = np.linspace(0, 1.1, yVal.shape[0])
    plt.title('Linear Regression')
    plt.xlabel('Data X (units)')
    plt.ylabel('Data Y (units)')
    fig = plt.figure()
    ax = fig.gca()
    yVal = linex*theta[0] + linex *theta[1]
    #ax.scatter(linex, [yVal])
    ax.scatter(xVal[:,1], xVal[:,2])


    #plt.plot(linex, liney.transpose(), color='red')
    #plt.show()

    # use your standRegress code from HW1  and show figure
    return theta


if __name__ == "__main__":
    xVal, yVal = loadDataSet('RRdata.txt')
    betaLR = ridgeRegress(xVal, yVal, lambdaV=0)
    print(betaLR)
    lambdaBest = cv(xVal, yVal)
    print(lambdaBest)
    betaRR = ridgeRegress(xVal, yVal, lambdaV=lambdaBest)
    print(betaRR)
    # depending on the data structure you use for xVal and yVal, the following line may need some change
    standRegress(xVal, yVal)
