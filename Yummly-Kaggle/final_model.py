import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import numpy as np
from data_processing import *


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
    test_X = test.drop("id", axis=1)

    clf = clf.fit(train_X, train_Y)
    forest_prob = pd.DataFrame(clf.predict_proba(test_X),
                                columns=clf.classes_)
    return pd.concat([test.id, forest_prob], axis=1), clf.classes_


if __name__ == "__main__":

    train, test = create_features(train_file ='data/train.json',
                                  test_file ='data/test.json')
    # Random Forest
    RF_preds, cls = fit_model(train, test,
                              clf_specs = {"kind":"RF",
                                           "n_estimators":400,
                                           'max_depth': 200})

    # Logistic Regression
    LR_preds, cls = fit_model(train, test,
                              clf_specs = {"kind":"LR", "C":2})

    # XGB Model
    XGB_preds, cls = fit_model(train, test,
                               clf_specs = {"kind":"XGB",
                                            "max_depth": 4,
                                            "learning_rate": 0.4,
                                            "n_estimators":400})

    test_id = test.id
    RF_preds = RF_preds.drop("id", axis=1)
    LR_preds = LR_preds.drop("id", axis=1)
    XGB_preds = XGB_preds.drop("id", axis=1)

    final_pred = (0.3*XGB_preds + 0.3*LR_preds + 0.4*LR_preds)

    test_pred = np.array(final_pred).argmax(axis=1)
    test_pred = cls[test_pred]

    # Copy the results to a pandas dataframe with an "id" column and
    # a "cusine" column
    output = pd.DataFrame( {"id":test_id, "cuisine":test_pred} )
    output[['id', 'cuisine']].to_csv( "submission_ensemble_avg.csv",
                                      index=False)

