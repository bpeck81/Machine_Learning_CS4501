# Machine Learning HW2-KNN

__author__ = 'bjp9pq'
import numpy as np
import operator, math
import time
#more importants
#from sklearn.neighbors import KNeighborsClassifier

#file is just a filename, this method read in file contents
# Att: there are many ways to read in one reference dataset,
# e.g., this template reads in the whole file and put it into one numpy array.
# (But in HW1, our template actually read the file into two numpy array, one for Xval, the other for Yval.
# Both ways are correct.)
def read_csv(file):
    data = np.genfromtxt(file, delimiter='\t', skip_header=1)
    np.random.shuffle(data)
    return data

#data is the full training numpy array
#k is the current iteration of cross validation
#kfold is the total number of cross validation folds
def fold(data, k, kfold):
    #50
    remainder = data.shape[0] % kfold
    fold_size = data.shape[0] // kfold
    start_row = fold_size *k
    additive  = 1 if k+1 <= remainder else 0
    start_row += k if k+1 <= remainder else remainder
    testing = data[start_row:start_row+fold_size+additive]
    training = data.copy()
    training = np.delete(training, np.s_[start_row:start_row+fold_size+additive], axis=0)
    return training, testing

def euclidian_distance(p1, p2):
    dist = 0
    for col in range(p1.size):
        dist += (p1[col] - p2[col])**2
    return math.pow(dist, .5)
#training is the numpy array of training data
#(you run through each testing point and classify based on the training points)
#testing is a numpy array, use this method to predict 1 or 0 for each of the testing points
#k is the number of neighboring points to take into account when predicting the label
def knn_vote(knn_list, test_rec):
    votes = {}
    for n in knn_list:
        classification = n[1][-1]
        if classification in votes:
            votes[classification] += 1
        else:
            votes[classification] = 1
    sorted_votes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def classify(training, testing, k):
    predictions = list()
    for test_rec in testing:
        knn_list = list()
        for train_rec in training:
            dist = euclidian_distance(test_rec, train_rec)
            if len(knn_list) == k:
                idx = -1
                for i in range(len(knn_list)):
                    if knn_list[i][0] > dist:
                        idx = i
                knn_list[idx] = (dist, train_rec) if idx != -1 else knn_list[idx]
            else:
                knn_list.append((dist, train_rec))
            # classification vote for record
        classification = knn_vote(knn_list, test_rec)
        predictions.append(classification)
    return np.asarray(predictions)

#predictions is a numpy array of 1s and 0s for the class prediction
#labels is a numpy array of 1s and 0s for the true class label
def calc_accuracy(predictions, labels):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            correct += 1
    x = float(correct / float(predictions.size))
    return x

def main():
    filename = "Movie_Review_Data.txt"
    kfold = 3
    k = str(input("Provide an odd k value: "))
    while(not k.isdigit()):
        k = input("Provide an odd k value: ")
    k = int(k)
    sum = 0
    start_time = time.time()
    data = np.asarray(read_csv(filename), dtype=float)
    for i in range(0, kfold):
        training, testing = fold(data, i, kfold)
        predictions = classify(training, testing, k)
        labels = testing[:,-1]
        sum += calc_accuracy(predictions, labels)
    accuracy = sum / kfold
    print(accuracy)


if __name__ == "__main__":
    main()