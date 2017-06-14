from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from data_processing import *


def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def get_classfier(clf_type = "RF"):
    if clf_type == "RF":
        return RandomForestClassifier()
    elif clf_type == "LR":
        return LogisticRegression()
    elif clf_type == "GB":
        return GradientBoostingClassifier()
    elif clf_type == "XGB":
        return XGBClassifier()


def cv_model(train, clf_type, param_lst, n_iter=5):
    clf = get_classfier(clf_type)
    train_Y = train.Y
    train_X = train.drop(["id", "Y"], axis=1)

    random_search = RandomizedSearchCV(clf,
                                       param_distributions=param_lst,
                                       n_iter = n_iter,
                                       cv=5)
    random_search.fit(train_X, train_Y)
    return random_search.cv_results_


if __name__ == "__main__":

    train, test = create_features(train_file ='data/train.json',
                                  test_file ='data/test.json')

    RF_param = {"n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [200, 300]}
    report(cv_model(train, clf_type="RF", param_lst = RF_param, n_iter=5))
    # Parameters: {'n_estimators': 500, 'max_depth': 200}
    # Mean validation score: 0.758(std: 0.002)

    LR_param = {"C": [2, 5, 10, 20, 30]}
    report(cv_model(train, clf_type="LR", param_lst = LR_param, n_iter=5))
    # Parameters: {'C': 2}
    # Mean validation score: 0.779(std: 0.002)

    XGB_param = {"max_depth": [1,2,3,4,5],
               "learning_rate": np.arange(0.1, 0.4, 0.1),
               "n_estimators": [200, 300, 400, 500]}
    report(cv_model(train, clf_type="XGB", param_lst=XGB_param, n_iter=5))
    # Parameters: {'n_estimators': 500, 'learning_rate': 0.20000000000000001, 'max_depth': 4}
    # Mean validation score: 0.789 (std: 0.001)

    ############################# CV MODEL OUTPUT #############################
    #
    # Model with rank: 1
    # Mean validation score: 0.765 (std: 0.005)
    # Parameters: {'n_estimators': 400, 'max_depth': 200}
    #
    # Model with rank: 2
    # Mean validation score: 0.764 (std: 0.004)
    # Parameters: {'n_estimators': 500, 'max_depth': 300}
    #
    # Model with rank: 3
    # Mean validation score: 0.764 (std: 0.004)
    # Parameters: {'n_estimators': 500, 'max_depth': 200}
    #
    # Model with rank: 4
    # Mean validation score: 0.763 (std: 0.006)
    # Parameters: {'n_estimators': 200, 'max_depth': 200}
    #
    # Model with rank: 5
    # Mean validation score: 0.761 (std: 0.005)
    # Parameters: {'n_estimators': 100, 'max_depth': 300}
    #
    # Model with rank: 1
    # Mean validation score: 0.779 (std: 0.005)
    # Parameters: {'C': 2}
    #
    # Model with rank: 2
    # Mean validation score: 0.777 (std: 0.005)
    # Parameters: {'C': 5}
    #
    # Model with rank: 3
    # Mean validation score: 0.774 (std: 0.004)
    # Parameters: {'C': 10}
    #
    # Model with rank: 4
    # Mean validation score: 0.771 (std: 0.005)
    # Parameters: {'C': 20}
    #
    # Model with rank: 5
    # Mean validation score: 0.768 (std: 0.005)
    # Parameters: {'C': 30}
    #
    # Model with rank: 1
    # Mean validation score: 0.794 (std: 0.004)
    # Parameters: {'n_estimators': 400, 'learning_rate': 0.40000000000000002, 'max_depth': 4}
    #
    # Model with rank: 2
    # Mean validation score: 0.789 (std: 0.004)
    # Parameters: {'n_estimators': 200, 'learning_rate': 0.30000000000000004, 'max_depth': 4}
    #
    # Model with rank: 3
    # Mean validation score: 0.784 (std: 0.004)
    # Parameters: {'n_estimators': 400, 'learning_rate': 0.30000000000000004, 'max_depth': 2}
    #
    # Model with rank: 4
    # Mean validation score: 0.778 (std: 0.005)
    # Parameters: {'n_estimators': 400, 'learning_rate': 0.20000000000000001, 'max_depth': 2}
    #
    # Model with rank: 5
    # Mean validation score: 0.751 (std: 0.004)
    # Parameters: {'n_estimators': 300, 'learning_rate': 0.10000000000000001, 'max_depth': 2}
    #
