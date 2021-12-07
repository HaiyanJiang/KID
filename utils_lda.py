#!/usr/bin/env python3
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.kernel_approximation import RBFSampler  # feature tranformation
from pyrfm.random_feature import RandomFourier, RandomKernel

import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold  # KFold,
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime

from ridger import PathSolver, coefs_plot


def _relabel(y):
    # Case 1: If the raw data is {0, 1} labeled, convert to {-1, 1} labeled.
    # Case 2: If the raw data 
    # If the raw label of y is {-1, 1}, converting it to {-2*n2/n, 2*n1/n}
    # where -2*n2/n corresponds to -1, and 2*n1/n corresponds to 1, 
    # n1 is the sample size of {-1} and n2 is the sample size of {1}.
    z = Counter(y)
    if len(z.keys()) != 2:
        raise ValueError('It is not a binary classification problem.')
    # {-1, 1}
    y1, y2 = sorted(list(z.keys()))
    n1, n2 = z[y1], z[y2]
    n = len(y)
    t = np.zeros(n, )
    t[y == y1] = -2 * n2/n
    t[y == y2] = 2 * n1/n
    return t


def _center(X, rowdim=True):
    if rowdim:
        # x1 = np.tile([1, 2, 3], (4, 1))
        # (x1.T - x1.mean(axis=1)).T
        return (X.T - X.mean(axis=1)).T
    else:
        return X - X.mean(axis=0)


def load_data(doc_name="data/Colon"):
    # dataname can take values of "Colon", "CNS", "Leukemia"
    # X, y = load_data(doc_name="data/Colon")
    import pickle
    with open(doc_name + '/features.pkl', 'rb') as f:
        features = pickle.load(f)
    with open(doc_name + '/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels)
    return features, labels


def rbf_transform_noact(X, ibase=1, gamma=1.0, random_state=1):
    # Equals the dimensionality of the computed feature space.
    d = X.shape[1]
    if int(ibase*d) % 2 != 0:
        d_2 = max(int(ibase*d) + 1, 2)  # is ibase=0.05, consider the minimum d_2.
    else:
        d_2 = max(int(ibase*d), 2)
    rbf_feature = RBFSampler(gamma=gamma, n_components=d_2, random_state=random_state)
    X_features = rbf_feature.fit_transform(X)
    return X_features


def rbf_transform(X, actfun='None', ibase=1, gamma=1.0, random_state=1):
    # Equals the dimensionality of the computed feature space.
    d = X.shape[1]
    if int(ibase*d) % 2 != 0:
        d_2 = max(int(ibase*d) + 1, 2)  # is ibase=0.05, consider the minimum d_2.
    else:
        d_2 = max(int(ibase*d), 2)
    rbf_feature = RBFSampler(gamma=gamma, n_components=d_2, random_state=random_state)
    X_features = rbf_feature.fit_transform(X)
    if actfun == 'None':
        return X_features
    x = torch.tensor(X_features)
    if actfun == "relu":
        X1 = torch.relu(x).data.numpy()
    elif actfun == "sigmoid":
        X1 = torch.sigmoid(x).data.numpy()
    elif actfun == "tanh":
        X1 = torch.tanh(x).data.numpy()
    elif actfun == "softplus":
        X1 = F.softplus(x).data.numpy()  # there's no softplus in torch
    else:
        X1 = torch.softmax(x, dim=0).data.numpy()
    return X1


def random_kernel_transform(X, actfun='None', ibase=1, random_state=1):
    # apply firstly the random fourier feature transformation, then relu().
    # X is n by d, should be dataframe. ibase is the enlarged size (augmented data).
    d = X.shape[1]
    if int(ibase*d) % 2 != 0:
        d_2 = max(int(ibase*d) + 1, 2)  # is ibase=0.05, consider the minimum d_2.
    else:
        d_2 = max(int(ibase*d), 2)
    # rf = RandomFourier(n_components=d_2, random_state=random_state)
    rf = RandomKernel(n_components=d_2, random_state=random_state)
    X_features = rf.fit_transform(X)
    if actfun == 'None':
        return X_features
    # ### use the activation functions in torch
    x = torch.tensor(X_features)
    # following are popular activation functions
    # y_relu = torch.relu(x).data.numpy()
    # y_sigmoid = torch.sigmoid(x).data.numpy()
    # y_tanh = torch.tanh(x).data.numpy()
    # y_softplus = F.softplus(x).data.numpy() # there's no softplus in torch
    # y_softmax = torch.softmax(x, dim=0).data.numpy()  
    # # softmax is a special kind of activation function, it is about probability
    if actfun == "relu":
        X1 = torch.relu(x).data.numpy()
    elif actfun == "sigmoid":
        X1 = torch.sigmoid(x).data.numpy()
    elif actfun == "tanh":
        X1 = torch.tanh(x).data.numpy()
    elif actfun == "softplus":
        X1 = F.softplus(x).data.numpy()  # there's no softplus in torch
    else:
        X1 = torch.softmax(x, dim=0).data.numpy()
        # # softmax is a special kind of activation function, it is about probability
    # print(X2.shape)
    return X1


def random_fourier_transform(X, actfun='None', ibase=1, random_state=1):
    # apply firstly the random fourier feature transformation, then relu().
    # X is n by d, should be dataframe. ibase is the enlarged size (augmented data).
    d = X.shape[1]
    if int(ibase*d) % 2 != 0:
        d_2 = max(int(ibase*d) + 1, 2)  # is ibase=0.05, consider the minimum d_2.
    else:
        d_2 = max(int(ibase*d), 2)
    rf = RandomFourier(n_components=d_2, random_state=random_state)
    # rf = RandomKernel(n_components=d_2, random_state=random_state)
    X_features = rf.fit_transform(X)
    if actfun == 'None':
        return X_features
    # ### use the activation functions in torch
    x = torch.tensor(X_features)
    # following are popular activation functions
    # y_relu = torch.relu(x).data.numpy()
    # y_sigmoid = torch.sigmoid(x).data.numpy()
    # y_tanh = torch.tanh(x).data.numpy()
    # y_softplus = F.softplus(x).data.numpy() # there's no softplus in torch
    # y_softmax = torch.softmax(x, dim=0).data.numpy()  
    # # softmax is a special kind of activation function, it is about probability
    if actfun == "relu":
        X1 = torch.relu(x).data.numpy()
    elif actfun == "sigmoid":
        X1 = torch.sigmoid(x).data.numpy()
    elif actfun == "tanh":
        X1 = torch.tanh(x).data.numpy()
    elif actfun == "softplus":
        X1 = F.softplus(x).data.numpy()  # there's no softplus in torch
    else:
        X1 = torch.softmax(x, dim=0).data.numpy()
        # # softmax is a special kind of activation function, it is about probability
    # print(X2.shape)
    return X1


class RawLDA():
    # ### never use this class.
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def fit(self, X, y):
        # X is n by p np.array, where n is the sample size and p is the dimension.
        # The y should be -1 and 1 otherwise the sign does not work.
        model = LinearDiscriminantAnalysis()
        model.fit(X, y)
        # s1, use the LDA score directly
        # model.score(X, y)  # mean accuracy
        print("LDA Accuracy of model score: ", model.score(X, y))  #  mean accuracy 
        y_0 = model.predict(X)
        print("LDA Accuracy of model predict: ", np.mean(y_0 == y))  #  mean accuracy 
        # s2, use the LDA coefficents and intercept to reconstruct the predicted value.
        w = (model.coef_).reshape(-1)
        b = model.intercept_
        # print("LDA intercept: ", b)
        y_w = np.sign(np.dot(X, w) + b)  # plus b.
        print("LDA reconstruct with w and b, Accuracy: ", np.mean(y_w == y))
        # s2.1, use the normalized LDA coefficents
        v = w / np.linalg.norm(w)
        y_v = np.sign(np.dot(X, v) + b)  # plus b.
        print("LDA reconstruct with normalized w and b, Accuracy: ", np.mean(y_v == y))
        # s3, use the LDA coefficents (the reduced dimension), and cg
        u1 = X[y == 1, :].mean(axis=0)
        u2 = X[y == -1, :].mean(axis=0)
        n1, n2 = np.sum(y == 1), np.sum(y == -1)
        c = -1/2 * np.dot(w.T, u1+u2) + np.log(n1/n2)
        y_2 = np.sign(np.dot(X, w) + c)
        print("LDA using w and c, Accuracy: ", np.mean(y_2 == y))
        cg = -1/2 * np.dot(w.T, u1 + u2)
        print("LDA center of the data: ", cg)
        y_21 = np.sign(np.dot(X, w) + cg)
        print("LDA using w and c, Accuracy: ", np.mean(y_21 == y))
        # s4, use the LDA normalized coefficents (the reduced dimension), and cg
        cg = -1/2 * np.dot(v.T, u1 + u2)
        y_3 = np.sign(np.dot(X, v) + cg)
        print("LDA using normalized w and c, Accuracy: ", np.mean(y_3 == y))
        acc = [
            model.score(X, y), np.mean(y_0 == y), np.mean(y_w == y), 
            np.mean(y_v == y), np.mean(y_2 == y), np.mean(y_3 == y)]
        self.accuracy = acc
        self.coef_ = (model.coef_).reshape(-1)
        # model score, model predict, reconstruct with w and b.
        # reconstruct with normalized w and b, w and c, normalized w and c.
        return acc


class IRLDA():
    # implicit Regularized LDA, which uses the SGD/GD to solve the relabeled OLS problem.
    # Here we only need to feed in the raw data, the relabeled proccess is done inside.
    def __init__(self, eta=0.5*1e-4, iter_max=1000, unitnorm=False, rowdim=False):
        self.eta = eta
        self.iter_max = iter_max
        self.unitnorm = unitnorm
        self.rowdim = rowdim
    
    def fit(self, X, y, strategy='OLS_batchSGD', cal_acc=False):
        # The X is np.array, y is labeled -1 and 1.
        self.strategy = strategy 
        solver = PathSolver(eta=self.eta, iter_max=self.iter_max, unitnorm=self.unitnorm)
        Xt = _center(X, self.rowdim)
        yt = _relabel(y)  # which is used in calculating the coefficients.
        if self.rowdim:
            if self.strategy == 'OLS_batchSGD':
                coefs = solver.OLS_batchSGD(Xt.T, yt, m=10)
            elif self.strategy == 'OLS_GD':
                coefs = solver.OLS_GD(Xt.T, yt)
        else:
            if self.strategy == 'OLS_batchSGD':
                coefs = solver.OLS_batchSGD(Xt, yt, m=10)
            elif self.strategy == 'OLS_GD':
                coefs = solver.OLS_GD(Xt, yt)
        self.coefs = coefs
        if cal_acc:
            u1 = X[y == 1, :].mean(axis=0)
            u2 = X[y == -1, :].mean(axis=0)
            n1, n2 = sum(y == 1), sum(y == -1)
            acc_list = []
            for i in range(self.iter_max):
                w = coefs[i]
                if np.linalg.norm(w) == 0:
                    v = w
                else:
                    v = w / np.linalg.norm(w)
                y_0 = np.sign(np.dot(X, w))
                y_1 = np.sign(np.dot(X, v))
                c_w = -1/2 * np.dot(w.T, u1 + u2) + np.log(n1/n2)
                y_2 = np.sign(np.dot(X, w) + c_w)
                c_v = -1/2 * np.dot(v.T, u1 + u2) + np.log(n1/n2)
                y_3 = np.sign(np.dot(X, v) + c_v)
                acc_list.append(
                    (np.mean(y_0 == y), np.mean(y_1 == y), np.mean(y_2 == y), np.mean(y_3 == y)))
                if i % 100 == 0:
                    print("IRLDA Accuracy using w: ", np.mean(y_0 == y))
                    print("IRLDA Accuracy using normalized w: ", np.mean(y_1 == y))
                    print("IRLDA Accuracy using w and c: ", np.mean(y_2 == y))
                    print("IRLDA Accuracy using normalized w and c: ", np.mean(y_3 == y))
            self.acc_list = acc_list
        return coefs
    
    def various_acc(self, X, y, w, u1, u2, n1=1, n2=1):
        if np.linalg.norm(w) == 0:
            v = w
        else:
            v = w / np.linalg.norm(w)
        y_0 = np.sign(np.dot(X, w))
        y_1 = np.sign(np.dot(X, v))
        c_w = -1/2 * np.dot(w.T, u1 + u2) + np.log(n1/n2)
        y_2 = np.sign(np.dot(X, w) + c_w)
        c_v = -1/2 * np.dot(v.T, u1 + u2) + np.log(n1/n2)
        y_3 = np.sign(np.dot(X, v) + c_v)
        acc = [
            np.mean(y_0 == y)*100 , np.mean(y_1 == y)*100, 
            np.mean(y_2 == y)*100, np.mean(y_3 == y)*100]
        return acc
    
    def fit_validate(self, X_train, y_train, X_val, y_val, strategy='OLS_batchSGD'):
        # X_train, y_train should come from the raw data X, y, not relabeled Xt, yt.
        self.strategy = strategy 
        solver = PathSolver(eta=self.eta, iter_max=self.iter_max, unitnorm=self.unitnorm)
        Xt = _center(X_train, self.rowdim)
        yt = _relabel(y_train)
        if self.rowdim:
            if self.strategy == 'OLS_batchSGD':
                coefs = solver.OLS_batchSGD(Xt.T, yt, m=10)
            elif self.strategy == 'OLS_GD':
                coefs = solver.OLS_GD(Xt.T, yt)
        else:
            if self.strategy == 'OLS_batchSGD':
                coefs = solver.OLS_batchSGD(Xt, yt, m=10)
            elif self.strategy == 'OLS_GD':
                coefs = solver.OLS_GD(Xt, yt)
        self.coefs = coefs
        z = Counter(y_train)
        if len(z.keys()) != 2:
            raise ValueError('It is not a binary classification problem.')
        # {-1, 1}
        y1, y2 = sorted(list(z.keys()))
        u1 = X_train[y_train == 1, :].mean(axis=0)
        u2 = X_train[y_train == -1, :].mean(axis=0)
        n1, n2 = sum(y_train == 1), sum(y_train == -1)
        acc_list = []
        for i in range(self.iter_max):
            w = self.coefs[i]
            train_acc = self.various_acc(X_train, y_train, w, u1, u2, n1, n2)
            test_acc = self.various_acc(X_val, y_val, w, u1, u2, n1, n2)
            acc_list.append(train_acc + test_acc)  # list +, as extend
        print("IRLDA Accuracy using w: ", train_acc[0])
        print("IRLDA Accuracy using normalized w: ", train_acc[1])
        print("IRLDA Accuracy using w and c: ", train_acc[2])
        print("IRLDA Accuracy using normalized w and c: ", train_acc[3])
        self.acc_list = acc_list
        return acc_list
    
    def cv_(self, X, y, strategy='OLS_batchSGD', n_splits=5, n_repeats=1, shuffle=True, random_state=123):
        # y should be labeled as -1 and 1.
        self.strategy = strategy 
        # solver = PathSolver(eta=self.eta, iter_max=self.iter_max, unitnorm=self.unitnorm)
        y_data = _relabel(y)
        cnames = [
            "Classifier", "n_features",  "Time Used", 
            "Train Accuracy w", "Train Accuracy normalized w", 
            "Train Accuracy w and c", "Train Accuracy normalized w and c", 
            "Val Accuracy w", "Val Accuracy normalized w", 
            "Val Accuracy w and c", "Val Accuracy normalized w and c"]
        log = pd.DataFrame(columns=cnames)
        name = self.strategy
        for i in range(n_repeats):
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state+i)
            for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):
                print(fold)
                ### Dividing data into folds
                X_train = X[train_index, :]
                y_train = y[train_index]
                y_fit = y_data[train_index]
                X_val = X[test_index, :]
                y_val = y[test_index]
                p = X_train.shape[1]
                a = datetime.now()
                # here we use the y_fit data to fit the model and get coefs.
                coefs = self.fit(X_train, y_fit, strategy=strategy, cal_acc=False)
                u1 = X_train[y_train == 1, :].mean(axis=0)  # y not y_data
                u2 = X_train[y_train == -1, :].mean(axis=0)
                n1, n2 = sum(y_train == 1), sum(y_train == -1)
                # # ### 
                # z = Counter(y_train)
                # if len(z.keys()) != 2:
                #     raise ValueError('It is not a binary classification problem.')
                # # {-1, 1}
                # y1, y2 = sorted(list(z.keys()))
                # u1 = X_train[y_train == y1, :].mean(axis=0)  # y not y_data
                # u2 = X_train[y_train == y2, :].mean(axis=0)
                # n1, n2 = sum(y_train == y1), sum(y_train == y2)
                # # ### 
                # acc_list = []
                for i in range(self.iter_max):
                    w = coefs[i]
                    train_acc = self.various_acc(X_train, y_train, w, u1, u2, n1, n2)
                    val_acc = self.various_acc(X_val, y_val, w, u1, u2, n1, n2)
                    # acc_list.append(train_acc + val_acc)  # list +, as extend
                    # acc_fold.append(acc_list)
                    b = datetime.now()
                    tt = (b - a).total_seconds()
                    log_entry = pd.DataFrame(
                        [[name, p, tt] + train_acc + val_acc], columns=cnames )
                    log = log.append(log_entry)
                print("IRLDA Accuracy using w: ", train_acc[0])
                print("IRLDA Accuracy using normalized w: ", train_acc[1])
                print("IRLDA Accuracy using w and c: ", train_acc[2])
                print("IRLDA Accuracy using normalized w and c: ", train_acc[3])
        scores = log["Val Accuracy w and c"]
        print('Mean Accuracy: %.3f (%.3f) of %s' % (np.mean(scores), np.std(scores), name))
        # log.mean(axis=0)
        return log
    




