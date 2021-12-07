#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold  # KFold,
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  # LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from datetime import datetime

# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, gamma='auto', probability=True),
    # NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]


def repeated_cv_clf(
        x_data, y_data, clf, n_splits=5, n_repeats=1, shuffle=True, random_state=123):
    log_cols=[
        "Classifier", "n_features", "Time Used", "Train Accuracy", "Train Log Loss",
        "Train AUC", "Val Accuracy", "Val Log Loss", "Val AUC"]
    log = pd.DataFrame(columns=log_cols)
    for i in range(n_repeats):
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state+i)
        for fold, (train_index, test_index) in enumerate(kfold.split(x_data, y_data)):
            ### Dividing data into folds
            X_train = x_data[train_index]
            y_train = y_data[train_index]
            X_val = x_data[test_index]
            y_val = y_data[test_index]
            p = X_train.shape[1]
            a = datetime.now()
            clf.fit(X_train, y_train)
            name = clf.__class__.__name__
            # print("="*30)
            # print(name)
            # print('****Results****')
            # ###### accuracy
            y_predictions = clf.predict(X_train)
            train_acc = accuracy_score(y_train, y_predictions)
            train_auc = roc_auc_score(y_train, y_predictions)
            # print("Train Accuracy: {:.4%}".format(train_acc))
            train_predictions = clf.predict_proba(X_train)
            train_ll = log_loss(y_train, train_predictions)
            # print("Train Log Loss: {}".format(train_ll))
            # ## to check the validation
            val_predictions = clf.predict(X_val)
            val_acc = accuracy_score(y_val, val_predictions)
            # print("Validation Accuracy: {:.4%}".format(val_acc))
            val_auc = roc_auc_score(y_val, val_predictions)
            # clf.score(X_val, y_val)
            # ###### log loss
            val_predictions = clf.predict_proba(X_val)
            val_ll = log_loss(y_val, val_predictions)
            # print("Validation Log Loss: {}".format(val_ll))
            b = datetime.now()
            tt = (b - a).total_seconds()
            log_entry = pd.DataFrame(
                [[name, p, tt, train_acc*100, train_ll, train_auc*100, 
                  val_acc*100, val_ll, val_auc*100]], 
                columns=log_cols)
            log = log.append(log_entry)
    scores = log["Val Accuracy"]
    print('Mean Accuracy: %.3f (%.3f) of %s' % (np.mean(scores), np.std(scores), name))
    # log.mean(axis=0)
    return log


def cv_clf_acc_loss(x_data, y_data, n_splits=5, n_repeats=1, shuffle=True, random_state=123):
    # Logging for Visual Comparison
    cols=[
        "Classifier", "n_features", "Time Used", "Train Accuracy", "Train Log Loss",
        "Train AUC", "Val Accuracy", "Val Log Loss", "Val AUC"]
    log = pd.DataFrame(columns=cols)
    for clf in classifiers:
        log_clf = repeated_cv_clf(
            x_data, y_data, clf, n_splits=n_splits, n_repeats=n_repeats, 
            shuffle=shuffle, random_state=random_state)
        log = log.append(log_clf)
        # # to save the log data.
        # name = clf.__class__.__name__
        # fname = 'acc_loss_%s.pkl' % (name)
        # with open(doc_name + '/' + fname, 'wb') as f:
        #     pickle.dump(log_clf, f)
    return log



