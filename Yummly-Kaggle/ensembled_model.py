import pandas as pd
import numpy as np
from data_processing import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit



def get_classfier(clf_specs = {"kind":"RF"}):
    if clf_specs['kind'] == "RF":
        return RandomForestClassifier(n_estimators = clf_specs['n_estimators'])
    elif clf_specs['kind'] == "LR":
        return LogisticRegression(C = clf_specs['C'])
    elif clf_specs['kind'] == "XGB":
        return XGBClassifier(max_depth=clf_specs['max_depth'],
                             learning_rate=clf_specs['learning_rate'],
                             n_estimators=clf_specs['n_estimators'])


def fit_model(train, test, clf_specs = {"kind":"RF"}):
    #Initilize a random forest classifier with 500 trees and
    #  fit it with the bag of words we created
    clf = get_classfier(clf_specs)
    train_Y = train.Y
    train_X = train.drop(["id", "Y"], axis=1)
    test_Y = test.Y.reset_index(drop=True)
    test_X = test.drop(["id", "Y"], axis=1)
    print test_X.shape

    clf = clf.fit(train_X, train_Y)
    forest_prob = pd.DataFrame(clf.predict_proba(test_X),
                                columns=clf.classes_)

    return pd.concat([test_Y, forest_prob], axis=1), clf.classes_

if __name__ == "__main__":

    train, test = create_features(train_file ='data/train.json',
                                  test_file ='data/test.json')

    # CREATE TRAIN AND VAL SET
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    for train_index, test_index in sss.split(train.drop(["id", "Y"], axis=1),
                                             train.Y):
        print("TRAIN:", train_index, "TEST:", test_index)

    sub_train = train.iloc[train_index]
    sub_test =  train.iloc[test_index]

    # Random Forest
    RF_preds, cls = fit_model(sub_train, sub_test,
                              clf_specs = {"kind":"RF",
                                           "n_estimators":400,
                                           'max_depth': 200})

    # Logistic Regression
    LR_preds, cls = fit_model(sub_train, sub_test,
                              clf_specs = {"kind":"LR", "C":2})

    # XGB Model
    XGB_preds, cls = fit_model(sub_train, sub_test,
                               clf_specs = {"kind":"XGB",
                                            "max_depth": 4,
                                            "learning_rate": 0.4,
                                            "n_estimators": 400})

    test_Y = sub_test.Y
    RF_preds = RF_preds.drop("Y", axis=1)
    LR_preds = LR_preds.drop("Y", axis=1)
    XGB_preds = XGB_preds.drop("Y", axis=1)

################################################################################
# Ensembling tuning on training
################################################################################

    # decide final weights based on test set performance
    acc_df = pd.DataFrame(columns=["i","j","k","acc"])

    wts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for i in wts:
        for j in wts:
            for k in wts:
                if abs(1 - (i + j + k) ) <= 0.01:
                    print abs(1 - (i + j + k) )
                    test_pred = (i*RF_preds + j*LR_preds + k*XGB_preds)/(i+j+k)
                    test_prediction = np.array(test_pred).argmax(axis=1)
                    test_prediction = cls[test_prediction]
                    acc = np.mean(test_prediction == test_Y)
                    acc_df = acc_df.append(pd.DataFrame({"i": i, "j":j, "k":k,
                                                         "acc":acc}, index=[0]))

    acc_df.sort(['acc'])


    # "i": 0.3, "j":0.3, "k":0.4


