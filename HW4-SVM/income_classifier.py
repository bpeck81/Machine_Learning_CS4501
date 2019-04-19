# Starting code for 4501 HW4 SVM
#WRITTEN IN PYTHON3
#bjp9pq
import numpy as np
from sklearn.svm import SVC
import random
import collections
import pandas as pd
import time
# Attention: You're not allowed to use the model_selection module in sklearn.
#            You're expected to implement it with your own code.
# from sklearn.model_selection import GridSearchCV

class SvmIncomeClassifier:
    def __init__(self):
        random.seed(0)

    def load_data(self, csv_fpath):
        col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                     'hours-per-week', 'native-country']
        col_names_y = ['label']
        headers = col_names_x
        headers.extend(col_names_y)
        numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                          'hours-per-week']
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                            'race', 'sex', 'native-country']

        # 1. Data pre-processing.
        # Hint: Feel free to use some existing libraries for easier data pre-processing.
        df = pd.read_csv(csv_fpath, header=None, names=headers, na_values='?')
        obj_df = df.select_dtypes(include=['object']).copy()
        obj_df[obj_df.isnull().any(axis=1)]
        lab_df = obj_df['label'].to_frame()
        del obj_df['label']
        for category in categorical_cols:
            #obj_df[category] = obj_df[category].astype('category')
            #obj_df[category] = obj_df[category].cat.codes
            obj_df = pd.get_dummies(obj_df, columns=[category])
        num_df = df.select_dtypes(exclude=['object']).copy()
        lab_df['label'] = lab_df['label'].astype('category')
        lab_df['label'] = lab_df['label'].cat.codes
        # obj_df.concat(num_df)
        # obj_df.concat(lab_df)
        combine = pd.concat([obj_df, num_df, lab_df], axis=1)
        if "native-country_ Holand-Netherlands" in combine:
            del combine["native-country_ Holand-Netherlands"]
        data = combine.as_matrix()
        np.random.seed(37)
        np.random.shuffle(data)
        x = data[:,:]
        y = data[:,-1]

        return x, y

    def split_data(self, xVal,yVal, k, kfold):
        yVal = yVal.reshape(yVal.shape[0], 1)
        data = np.hstack((xVal, yVal))
        remainder = data.shape[0] % kfold
        fold_size = data.shape[0] // kfold
        start_row = k * fold_size
        additive = 1 if k + 1 <= remainder else 0
        start_row += k if k + 1 <= remainder else remainder
        testing = data[start_row:start_row + fold_size + additive]
        training = data.copy()
        training = np.delete(training, np.s_[start_row:start_row + fold_size + additive], axis=0)
        return training, testing
    def calc_error(self, x, y):
        corr = 0
        for i in range(len(x)):
            if x[i] == y[i]:
                corr+=1
        j = corr/float(len(x))
        print(j)
        return j
    def train_and_select_model(self, training_csv):
        x_train, y_train = self.load_data(training_csv)

        # 2. Select the best model with cross validation.
        # Attention: Write your own hyper-parameter candidates.
        param_set = [
            #{'kernel': 'rbf', 'C': 1, 'degree': 1},
            #{'kernel': 'rbf', 'C': .01, 'degree': 1},
            #{'kernel': 'linear', 'C': 1, 'degree': 1},
            #{'kernel': 'linear', 'C': .01, 'degree': 1},
            #{'kernel': 'poly', 'C': 1, 'degree': 2},
            {'kernel': 'sigmoid', 'C': 1, 'degree': 1},
        ]
        kfold = 3
        preds = []
        for params in param_set:
            sum_accuracy = 0
            p = []
            for k in range(kfold):
                training, testing = self.split_data(x_train, y_train, k, kfold)
                x_training = training[:,:-1]
                y_training = training[:,-1]
                x_testing = testing[:,:-1]
                y_testing = testing[:,-1]
                clf = SVC(C=params['C'], kernel=params['kernel'])
                start = time.time()
                clf.fit(x_training, y_training)
                predictions = clf.predict(x_testing)
                end = time.time()
                print(end-start)
                score = clf.score(x_testing, y_testing)
                score1 = clf.score(x_training, y_training)
                end = time.time()
                #print(end-start)
                print('K=' + str(k) +" Testing score="+ str(score))
                print('K=' + str(k) +" Training score="+ str(score1))
                sum_accuracy += score
                #p.extend(predictions)
            accuracy = sum_accuracy/float(kfold)
            print(accuracy)
            preds.append((accuracy, params))
            #assumes error is different and if same predicted values are deemed equally correct
        sort =  sorted(preds, key=lambda x: x[0])
        print(sort)
        best_model = sort[0][1]
        clf = SVC(C=best_model['C'], kernel=best_model['kernel'], degree=best_model['degree'])
        clf.fit(x_train, y_train)
        best_model = clf
        best_score = sort[0][0]
        return best_model, best_score

    def predict(self, test_csv, trained_model):
        x_test, _ = self.load_data(test_csv)
        predictions = trained_model.predict(x_test)
        return predictions

    def output_results(self, predictions):
        # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
        # Hint: Don't archive the files or change the file names for the automated grading.
        with open('predictions.txt', 'w') as f:
            for pred in predictions:
                if pred == 0:
                    f.write('<=50K\n')
                else:
                    f.write('>50K\n')

if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    clf = SvmIncomeClassifier()
    trained_model, cv_score = clf.train_and_select_model(training_csv)
    print("The best model was scored %.2f" % cv_score)
    predictions = clf.predict(testing_csv, trained_model)
    clf.output_results(predictions)


