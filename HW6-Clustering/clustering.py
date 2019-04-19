#!/usr/bin/python
#Uses Python3
#bjp9pq
#Brandon Peck
import sys, operator
import numpy as np
import math
import matplotlib.pyplot as plt

def loadData(fileDj):
    data = np.genfromtxt(fileDj)
    return data

## K-means functions

def getInitialCentroids(X, k):
    initialCentroids = [[] for i in range(k)]
    for col in range(X.shape[1]):
        for i in range(k):
            rand = np.random.randint(np.min(X[:,col]), np.max(X[:,col]))
            initialCentroids[i].append(rand)
    return initialCentroids

def getDistance(pt1,pt2):
    dist = 0
    domain = min(len(pt1), len(pt2))
    for i in range(domain):
        dist += (pt2[i] - pt1[i])**2
#    dist = math.sqrt(dist)
    return dist

def allocatePoints(X,clusters):
    members = [[] for i in range(len(clusters))]
    for p in range(X.shape[0]):
        min = sys.maxsize
        j = 0
        for i in range(len(clusters)):
            dist = getDistance(X[p,:-1], clusters[i])
            if dist < min:
                min = dist
                j = i
        if j ==0:
            pass
        members[j].append(X[p,:])
    return members

def updateCentroids(centroids, clusters):
    point_len = 0
    for i in range(len(clusters)):
        if len(clusters[i]) != 0:
            point_len = len(clusters[i][0])
            break

    new_centroids = [0 for i in range(len(centroids))]
    for i in range(len(clusters)):
        avg_point = [0 for j in range(point_len)]
        for point in clusters[i]:
            for r in range(len(point)):
                avg_point[r] += point[r]
        avg_point = [p/len(clusters[i]) for p in avg_point]
        new_centroids[i] = avg_point
    return new_centroids


def visualizeClusters(clusters):
    colors = [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]
    for i in range(len(clusters)):
        X = np.array(clusters[i])
        color = colors[i]
        #assumes 2d data
        plt.scatter(X[:,0], X[:,1], c=color)
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.show()

def kmeans(X, k, maxIter=1000):
#    X = X[:,-1]
    centroids = getInitialCentroids(X,k)
    clusters =  [[] for i in range(len(centroids))]
    for i in range(maxIter):
        clusters = allocatePoints(X, centroids)
        new_cents = updateCentroids(centroids, clusters)
        if new_cents == centroids: break
        else: centroids = new_cents
    return clusters, centroids

def obj_function(clusters, centroids):
    dist = 0
    for i in range(len(clusters)):
        for point in clusters[i]:
            #exclude y value in distance calculation
            point = point[:-1]
            dist += getDistance(point, centroids[i])
    return dist

def kneeFinding(X,kList):
    obj = []
    for k in kList:
        clusters, centroids = kmeans(X, k)
        #visualizeClusters(clusters)
        obj.append(obj_function(clusters, centroids))
    plt.scatter(kList, obj)
    plt.plot(kList, obj)
    plt.xlabel('K')
    plt.ylabel('Objective Function ')
    plt.show()
    pass

def purity(X, clusters):
    purities = []
    for i in range(len(clusters)):
        counts = {}
        for j in range(len(clusters[i])):
            y = clusters[i][j][-1]
            if y in counts:
                counts[y] += 1
            else:
                counts[y] = 1
        s = sorted(counts.items(), key=operator.itemgetter(1))
        purities.append(s[-1][1]/float(len(clusters[i])))
    return purities

def main():
    #######dataset path
    datadir = sys.argv[1]
    pathDataset1 = datadir+'/humanData.txt'
    pathDataset2 = datadir+'/audioData.txt'
    dataset1 = loadData(pathDataset1)
    dataset2 = loadData(pathDataset2)

    #Q4
    kneeFinding(dataset1,range(1,7))

    #Q5
    clusters, c = kmeans(dataset1, 2, maxIter=1000)
    visualizeClusters(clusters)
    p = purity(dataset1,clusters)
    print(p)


if __name__ == "__main__":
    main()