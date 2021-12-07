#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
plt.rcParams['font.size'] = 18

from collections import Counter
from datetime import datetime
import pickle


from utils_lda import rbf_transform, random_fourier_transform, random_kernel_transform
from utils_lda import load_data, _center,_relabel, IRLDA

from cv_clf import cv_clf_acc_loss
# cv_clf_acc_loss(x_data, y_data)


def test_cv_IRLDA():
    X, y = load_data(doc_name="data/Leukemia")
    X = np.array(X)
    y[y != 1] = -1
    X1 = _center(X, rowdim=False)  # X is n by p, y is 0-1
    y1 = _relabel(y)
    print(X.shape)
    # model1: the traditional LDA
    
    ilda = IRLDA(eta=0.5*1e-3, iter_max=1000, unitnorm=False)
    coefs = ilda.fit(X, y, strategy='OLS_batchSGD', cal_acc=True)
    
    from sklearn.model_selection import StratifiedKFold  # KFold,
    kfold = StratifiedKFold(n_splits=5,  shuffle=True, random_state=123)
    for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):
        print(fold)
        ### Dividing data into folds
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_val = X[test_index, :]
        y_val = y[test_index]
        a = datetime.now()
    
    acc_list = ilda.fit_validate(X_train, y_train, X_val, y_val, strategy='OLS_batchSGD')
    
    log = ilda.cv_(X, y, strategy='OLS_batchSGD', n_splits=5, n_repeats=1, shuffle=True, random_state=123)
    # the total length is iter_max * (n_splits * n_repeats)


def main():
    # random fourier features, 'rff'
    # doc_root="data/Colon"; transformation="rff"
    # doc_root="data/CNS"; transformation="rff"
    # doc_root="data/Leukemia"; transformation="rff"
    # doc_root="data/Colon"; transformation="rbf"
    # doc_root="data/CNS"; transformation="rbf"
    # doc_root="data/Leukemia"; transformation="rbf"
    # doc_root="data/Leukemia"; transformation="rbf"; transformation="rff";
    # doc_root="data/LSVT"; transformation="rbf"; transformation="rff";
    from args import args
    doc_root = args.doc_root 
    transformation = args.transformation
    actfun = args.actfun
    
    if actfun == 'None':
        # the document for the results.
        doc_name = doc_root + "/Results"
    else:
        doc_name = doc_root + "/Results_%s" %(actfun)
    # the document for the results.
    if not os.path.exists(doc_name):
        print("NO this dir and make a new one.")
        os.makedirs(doc_name)
    # # load the data
    X, y = load_data(doc_root)
    X = np.array(X)
    # y[y == 0] = (-1)
    y[y != 1] = (-1)  # binary classification, must be -1 and 1.
    
    ##### use the raw data ##### 
    log_clf = cv_clf_acc_loss(X, y, n_splits=5, n_repeats=1, shuffle=True, random_state=123)
    fname = 'Classifier_%s_ibase_0.pkl' %(transformation)
    with open(doc_name + '/' + fname, 'wb') as f:
        pickle.dump(log_clf, f)
    irlda = IRLDA(eta=0.5*1e-3, iter_max=1000, unitnorm=False)
    log_sgd = irlda.cv_(X, y, strategy='OLS_batchSGD', n_splits=5, n_repeats=1, shuffle=True, random_state=123)
    # the total length of log_irlda is iter_max * (n_splits * n_repeats).
    fname = 'IRLDA_SGD_%s_ibase_0.pkl' %(transformation)
    with open(doc_name + '/' + fname, 'wb') as f:
        pickle.dump(log_sgd, f)
    log_gd = irlda.cv_(X, y, strategy='OLS_GD', n_splits=5, n_repeats=1, shuffle=True, random_state=123)
    # the total length of log_irlda is iter_max * (n_splits * n_repeats).
    fname = 'IRLDA_GD_%s_ibase_0.pkl' %(transformation)
    with open(doc_name + '/' + fname, 'wb') as f:
        pickle.dump(log_gd, f)
    ##### use the raw data #####
    # # # ## a small modification.
    # if doc_root == "data/Leukemia":
    #     print("YES! Leukemia!")
    #     alist = np.arange(23, 50, 1)
    # else:
    #     alist = np.arange(1, 50, 1)
    # # ## a small modification.
    # ## feature enhancement
    alist = np.arange(1, 50, 1); addraw = True  # sol_add
    # alist = np.arange(1, 50, 1); addraw = False  # sol_only, with only rbf
    # alist = np.arange(0.05, 1.01, 0.05); addraw = True  #  reduction_add
    # alist = np.arange(0.05, 1.01, 0.05); addraw = False  #  reduction_only
    irlda = IRLDA(eta=0.5*1e-3, iter_max=1000, unitnorm=False)
    for i in alist:
        # ### choose the transform function.
        if transformation == 'rbf':
            X1 = rbf_transform(X, actfun=actfun, ibase=i, gamma=1.0, random_state=int(1+i*100))
        elif transformation == 'rkt':
            X1 = random_kernel_transform(X, actfun=actfun, ibase=i, random_state=int(1+i*100))
        elif transformation == 'rft':
            X1 = random_fourier_transform(X, actfun=actfun, ibase=i, random_state=int(1+i*100))
        # X1 = random_fourier_transform(X, actfun="sigmoid", ibase=i, random_state=int(1+i*100))
        # X1 = rbf_transform(X, ibase=i, gamma=1.0, random_state=int(1+i*100))
        if addraw:
            X2 = pd.concat([pd.DataFrame(X), pd.DataFrame(X1)], axis=1)
            X2 = np.array(X2)
            tail_name = '%s_ibase_%s.pkl' % (transformation, "{:.0f}".format(i+1) if i >= 1 else "{:.2f}".format(i))
        else:
            X2 = X1
            tail_name = '%s_ibase_%s.pkl' % (transformation, "{:.0f}".format(i+1) if i > 1 else "{:.2f}".format(i))
        log_clf = cv_clf_acc_loss(X2, y, n_splits=5, n_repeats=1, shuffle=True, random_state=123)
        # tail_name = '%s_ibase_%s.pkl' % (transformation, "{:.0f}".format(i+1) if i > 1 else "{:.2f}".format(i))
        # tail_name = '%s_ibase_%s.pkl' % (transformation, "{:.2f}".format(i) if i <= 1 else "{:.0f}".format(i+1))
        fname = 'Classifier_' + tail_name
        # fname = 'Classifier_%s_ibase_%s.pkl' % (transformation, i)
        with open(doc_name + '/' + fname, 'wb') as f:
            pickle.dump(log_clf, f)
        # coefs = ilda.fit(X, y, strategy='OLS_batchSGD', cal_acc=True)
        # acc_list = ilda.fit_validate(X_train, y_train, X_val, y_val, strategy='OLS_batchSGD')
        log_sgd = irlda.cv_(
            X2, y, strategy='OLS_batchSGD', n_splits=5, n_repeats=1, shuffle=True, random_state=123)
        # # the total length is iter_max * (n_splits * n_repeats)
        # fname = 'IRLDA_SGD_%s_ibase_%s.pkl' %(transformation, i)
        fname = 'IRLDA_SGD_' + tail_name
        with open(doc_name + '/' + fname, 'wb') as f:
            pickle.dump(log_sgd, f)
        # ### the gd
        log_gd = irlda.cv_(
            X2, y, strategy='OLS_GD', n_splits=5, n_repeats=1, shuffle=True, random_state=123)
        # # the total length is iter_max * (n_splits * n_repeats)
        # fname = 'IRLDA_GD_%s_ibase_%s.pkl' %(transformation, i)
        fname = 'IRLDA_GD_' + tail_name
        with open(doc_name + '/' + fname, 'wb') as f:
            pickle.dump(log_gd, f)


if __name__ == '__main__':
    print("YES!")
    # main()
    

