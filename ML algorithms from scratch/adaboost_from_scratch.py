"""
A class for Adaboost is Python from scratch. The results on a random 
dataset were compared with the implementation of Adaboost in 
scikit-learn package and the results were found to match exactly.
"""

from __future__ import division
from numpy import *
import numpy as np
from sklearn.datasets import make_hastie_10_2
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from scipy import stats
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class AdaBoost:

    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.N = len(self.Y_train)
        self.weights = ones(self.N)/self.N
        self.ALPHA = []
        self.clfs = []

    def fitted(self, n_estimators):
        # getting predictions
        self.n_estimators = n_estimators
        for _ in range(n_estimators):
            clf = DecisionTreeClassifier(max_depth=1)
            CLF = clf.fit(self.X_train, self.Y_train,
                            sample_weight=self.weights)
            preds = CLF.predict(self.X_train)
            errors = array([preds[i]!=self.Y_train[i] for i in range(self.N)])
            e = np.sum(errors*self.weights) / np.sum(self.weights)
            if e >= 0.5: break
            alpha = np.log((1-e)/e)
            # print 'e=%f alpha=%f'%(e, alpha)
            w = zeros(self.N)
            for i in range(self.N):
                if errors[i] == 1: w[i] = self.weights[i] * exp(alpha)
                else: w[i] = self.weights[i]
            self.weights = w #/ w.sum()
            self.ALPHA.append(alpha)
            self.clfs.append(CLF)

    def predict(self, test_X):
        y = np.zeros(len(test_X))
        for (alpha, clf) in zip(self.ALPHA, self.clfs):
            y += alpha*clf.predict(test_X)
        self.signed_preds = sign(y)
        return self.signed_preds

    def evaluate(self, Y):
        n = len(Y)
        errors = array([self.signed_preds[i]!=Y[i]
                        for i in range(n)])
        return errors

    def plot_error(self, test_X, Y):
        n = len(Y)
        y = np.zeros(len(test_X))
        lst = []
        for (alpha, clf) in zip(self.ALPHA, self.clfs):
            y += alpha*clf.predict(test_X)
            preds = sign(y)
            err = np.mean(array([preds[i]!=Y[i] for i in range(n)]))
            lst.append(err)
        return lst



if __name__ == '__main__':

    #generating dataset
    features = pd.DataFrame(np.random.randn(2000, 10))
    features['chisq_rv']= np.square(features).sum(axis=1)
    threshold = chi2.ppf(0.5, 10)
    features['Y'] = np.where(features['chisq_rv']>threshold, 1, -1)

    X_train = features.drop(['chisq_rv', 'Y'], axis=1).values
    Y_train = np.where(features['chisq_rv']>threshold, 1, -1)

    features = pd.DataFrame(np.random.randn(10000, 10))
    features['chisq_rv']= np.square(features).sum(axis=1)
    threshold = chi2.ppf(0.5, 10)
    features['Y'] = np.where(features['chisq_rv']>threshold, 1, -1)

    X_test = features.drop(['chisq_rv', 'Y'], axis=1).values
    Y_test = np.where(features['chisq_rv']>threshold, 1, -1)

    clf = DecisionTreeClassifier(max_depth=1)
    m = AdaBoost(X_train, Y_train)
    m.fitted(400)

    print m.predict(X_train)
    print  "Train Error with my Adaboost is ", m.evaluate(Y_train).mean()
    print m.predict(X_test)
    print  "Test Error with my Adaboost is ", m.evaluate(Y_test).mean()

    err_tr = m.plot_error(X_train, Y_train)
    m.plot_error(X_train, Y_train)
    plt.plot(range(1,401), err_tr, color='peru')

    err_tst = m.plot_error(X_test, Y_test)
    plt.plot(range(1,401), err_tst, color='tomato')

    dTree_Score = 1 - clf.fit(X_train, Y_train).score(X_train, Y_train)
    plt.plot(range(1,401), np.tile(dTree_Score,400), '--', color='black')
    plt.text(250, .465, 'Single Stump')

    plt.xlabel('Boosting Iterations')
    plt.ylabel('Test Error')
    peru_patch = mpatches.Patch(color='peru', label='Train Error', linestyle = 'solid')
    tomato_patch = mpatches.Patch(color='tomato', label='Test Error')
    plt.legend(handles=[peru_patch, tomato_patch], loc='upper right', prop={'size':10})
    plt.show()

    # comparing with sklearn solution
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=400).fit(X_train, Y_train)
    # prediction accuracy
    print "Train Error with sklearn Adaboost is ", 1-bdt.score(X_train, Y_train)
    print "Test Error with sklearn Adaboost is ", 1-bdt.score(X_test, Y_test)


