#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def readBALANCEdata():
    df = pd.read_csv('balance-scale.data', header=None)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    #convert target to numerical values
    y = np.where(y == 'L', 0, np.where(y == 'R', 1, np.where(y == 'B', 2, 3)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    return X_train, y_train, X_val, y_val, X_test, y_test

def KNNTraining(X_train, y_train, X_val, y_val, X_test, y_test):
    #reshape target to be 1D array for KNN
    y_train = y_train.reshape((len(y_train), ))
    best_neigh = 0
    max_acc = 0
    val_accs = []
    #better to choose neighbors to be odd numbers to avoid ties
    neighbors = [1, 3, 5, 8, 9, 11, 13, 15, 18, 20, 21, 24, 25]
    for neighbor in neighbors:
        model = KNeighborsClassifier(n_neighbors=neighbor)
        #k-fold cross validation
        val_acc = np.mean(cross_val_score(model, X_train, y_train, cv=7, scoring='accuracy'))
        val_accs.append(val_acc)
        model.fit(X_train, y_train)
        pred = model.predict(X_train)
        if val_acc > max_acc:
            best_neigh = neighbor
            max_acc = val_acc

    model = KNeighborsClassifier(n_neighbors=best_neigh)
    performance = test_accuracy(model, X_train, y_train, X_test, y_test)
    print('Best number of neighbors = ', best_neigh)
    print('KNN Accuracy in percent = ',performance)
     #plotting
    plot(neighbors, val_accs,'Neighbors', 'Cross Validation Accuracy','KNN Validation Accuracy vs. Neighbors',"KNN.jpg")

def SVMTraining(X_train, y_train, X_test, y_test):
    y_train = y_train.reshape((len(y_train),))
    hyperparameters = [0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]
    best_hyper = 0
    validation_accuracy = []
    max_accuracy = 0   
    loss = []

    for hyper in hyperparameters:
        svm = SVC(C=hyper)
        # K-fold cross validation
        accuracy = np.mean(cross_val_score(svm, X_train, y_train, cv=10, scoring='accuracy'))
        validation_accuracy.append(accuracy)
        #training loss which is the hinge loss here
        svm.fit(X_train, y_train)
        decision_values = svm.decision_function(X_train)
        predicted_labels = np.argmax(decision_values, axis=1)
        hinge_loss = np.maximum(0, 1 - y_train * decision_values[np.arange(len(y_train)), predicted_labels])
        training_loss = hinge_loss.mean()
        loss.append(training_loss)
  
        if accuracy > max_accuracy:
            best_hyper = hyper
            max_accuracy = accuracy

    model = SVC(C=best_hyper)
    test_acc = test_accuracy(model, X_train, y_train, X_test, y_test)
    print('Best hyperparameter is', best_hyper)
    print('SVM test accuracy is', test_acc)
    plot(hyperparameters, validation_accuracy, 'C Value', 'Cross Validation Accuracy','SVM Validation Accuracy vs Hyperparameters',"SVM.jpg")

def LogesticRegressionTraining(X_train, y_train, X_val, y_val, X_test, y_test):
    alphas = [0.0001,0.001,0.01,0.1]
    best_alpha = 0
    validation_accuracy = []
    l1_acc = []
    l2_acc = []
    elastic_acc = []
    max_accuracy = 0
    best_penalty = ''
    for epoch in range(10):
        for penalty in ['l1', 'l2', 'elasticnet']:
            for alpha in alphas:
                model = SGDClassifier(loss='log_loss', penalty=penalty, alpha=alpha, max_iter=1000)
                model.fit(X_train, y_train)
                # K-fold cross validation
                accuracy = np.mean(cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10))
                validation_accuracy.append(accuracy)
                if penalty == 'l1':
                    l1_acc.append(accuracy)
                elif penalty == 'l2':
                    l2_acc.append(accuracy)
                else:
                    elastic_acc.append(accuracy)
                if accuracy > max_accuracy:
                    best_alpha = alpha
                    best_penalty = penalty
                    max_accuracy = accuracy
   
    model = SGDClassifier(alpha=best_alpha, loss='log_loss', penalty=best_penalty,  max_iter=1000)
    test_acc = test_accuracy(model, X_train, y_train, X_test, y_test)
    print('Best hyperparameter is', best_alpha)
    print('Best penalty is', best_penalty)
    print('Logistic Regression test accuracy is', test_acc)

    # print(validation_accuracy)
    # print(l1_acc)
    # print(l2_acc)
    # print(elastic_acc)

    plt.plot( l1_acc, label='l1')
    plt.plot( l2_acc, label='l2')
    plt.plot( elastic_acc, label='elasticnet')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Validation Accuracy')
    plt.title('Logistic Regression Validation Accuracy ')
    plt.legend()
    plt.savefig('LogisticRegression.jpg')

def test_accuracy(model,X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, pred)
    return test_acc

def plot(hyperparameters,accuracy,xlabel,ylabel,title,fileName):
    plt.plot(hyperparameters, accuracy)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(fileName)
    plt.cla()

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = readBALANCEdata()
    KNNTraining(X_train, y_train, X_val, y_val, X_test, y_test)
    SVMTraining(X_train, y_train, X_test, y_test)
    LogesticRegressionTraining(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == '__main__':
    main()



            