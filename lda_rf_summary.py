#!/usr/bin/env python3
import os
import glob
import pickle
import pandas as pd
import numpy as np
import re


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


def classifier_result_summary(doc_root, strategy, transformation):
    # strategy = 'Classifier'; transformation = 'rff'
    # doc_root = "data/CNS"; doc_root = "data/Colon"; doc_root="data/LSVT"
    # get the results.
    doc_name = doc_root + "/ResultsNew"
    if strategy == 'Classifier':
        glob_filename = doc_name + '/%s_%s_ibase_*.pkl' %(strategy, transformation)
    else:
        glob_filename = doc_name + '/IRLDA_%s_%s_ibase_*.pkl' %(strategy, transformation)
    file_list = glob.glob(glob_filename)
    ilist = sorted([float(f[:-4].split('_')[-1]) for f in file_list])
    # filename = 'data/PGD/ResultsNew/Classifier_rbf_ibase_0.30.pkl'
    # float(filename[:-4].split('_')[-1])  # split with '_' and '.'
    
    # print(len(file_list))
    filename = file_list[0]
    with open(filename, 'rb') as f:
        log = pickle.load(f)
    # print(log.columns)
    log_cols = [
        'Time Used', 'Train Accuracy', 'Train Log Loss', 'Train AUC', 
        'Val Accuracy', 'Val Log Loss', 'Val AUC']
    cnames_1 = [ele + ' Mean' for ele in log_cols]
    cnames_2 = [ele + ' Std' for ele in log_cols]
    head_names = ['Classifier', 'n_features', 'n_bases']
    # log.columns
    cnames = head_names + cnames_1 + cnames_2
    clf_names = list(np.unique(log['Classifier']))
    dt = {clf: pd.DataFrame(columns=cnames) for clf in clf_names}
    
    for filename in file_list:
        with open(filename, 'rb') as f:
            log = pickle.load(f)
        # the result
        dfmean = log.groupby(['Classifier']).apply(np.mean).reset_index(drop=False)
        dfstd = log.groupby(['Classifier']).apply(np.std).reset_index(drop=False)
        nfeat = int(list(set(dfmean['n_features']))[0])
        ibase = float(filename[:-4].split('_')[-1])
        for clf in clf_names:
            df = dt[clf]
            vmean = dfmean[dfmean['Classifier'] == clf].drop(['Classifier', 'n_features'], axis=1)
            vstd = dfstd[dfstd['Classifier'] == clf].drop(['Classifier', 'n_features'], axis=1)
            vres = vmean.values.reshape(-1,).tolist() + vstd.values.reshape(-1,).tolist()
            df_entry = pd.DataFrame([[clf, nfeat, ibase] + vres], columns=cnames)
            df = df.append(df_entry)
            dt[clf] = df
    
    # clf_list = [
    #     'AdaBoostClassifier',
    #     'DecisionTreeClassifier',
    #     'GaussianNB',
    #     'GradientBoostingClassifier',
    #     'KNeighborsClassifier',
    #     'LinearDiscriminantAnalysis',
    #     'QuadraticDiscriminantAnalysis',
    #     'RandomForestClassifier',
    #     'SVC']
    # df = dt['AdaBoostClassifier']
    doc_save = doc_root + "/ResultSummary"
    if not os.path.exists(doc_save):
        print("NO this dir and make a new one.")
        os.makedirs(doc_save)
    fname = '%s_%s.pkl' %(strategy, transformation)
    with open(doc_save + '/' + fname, 'wb') as f:
        pickle.dump(dt, f)
    
    # from matplotlib import pylab as plt
    # for clf in clf_names:
    #     df = dt[clf]
    #     df = df.sort_values('n_bases').reset_index(drop=True)
    #     x = df['Val Accuracy Mean']
    #     from matplotlib import pylab as plt
    #     plt.plot(df['n_features'], df['Val Accuracy Mean'])
    #     plt.legend()


def IRLDA_result_summary(doc_root, strategy, transformation):
    
    def equal_consecutive(x, k=100, tol=1e-5):
        return all(abs(np.diff(x[-(k):])) < tol) if len(x) >= k else  False
    # doc_root = "data/CNS"; strategy = 'SGD'; transformation = 'rbf'
    doc_save = doc_root + "/ResultSummary"
    if not os.path.exists(doc_save):
        print("NO this dir and make a new one.")
        os.makedirs(doc_save)
    # get the results.
    doc_name = doc_root + "/ResultsNew"
    if strategy == 'Classifier':
        glob_filename = doc_name + '/%s_%s_ibase_*.pkl' %(strategy, transformation)
    else:
        glob_filename = doc_name + '/IRLDA_%s_%s_ibase_*.pkl' %(strategy, transformation)
    
    file_list = glob.glob(glob_filename)
    filename = file_list[0]
    with open(filename, 'rb') as f:
        log = pickle.load(f)
    log.columns
    log_cols = [
        'Time Used', 'Train Accuracy w', 'Train Accuracy normalized w', 
        'Train Accuracy w and c', 'Train Accuracy normalized w and c', 
        'Val Accuracy w', 'Val Accuracy normalized w',
        'Val Accuracy w and c', 'Val Accuracy normalized w and c']
    cnames_1 = [ele + ' Mean' for ele in log_cols]
    cnames_2 = [ele + ' Std' for ele in log_cols]
    head_names = ['Classifier', 'n_iters', 'n_features', 'n_bases']
    # log.columns
    cnames = head_names + cnames_1 + cnames_2
    dt1 = pd.DataFrame(columns=cnames)
    dt2 = pd.DataFrame(columns=cnames)
    
    clf = np.unique(log['Classifier']).tolist()[0]
    itmax = 1000
    
    for filename in file_list:
        with open(filename, 'rb') as f:
            log = pickle.load(f)
        # rec = re.compile(r'\d+')
        # ibase = int(rec.findall(filename)[0])
        nfeat = int(list(set(log['n_features']))[0])
        ibase = float(filename[:-4].split('_')[-1])
        
        df_list = []
        for k in range(5):
            df = log[(itmax*k):(itmax*(k+1))].reset_index(drop=True)
            df = df.reset_index(drop=False)  # adding one column for groupby.
            df_list.append(df)
        # the result
        data = pd.concat(df_list)
        dfmean = data.groupby(['index']).apply(np.mean).reset_index(drop=True).drop(['index'], axis=1)
        dfstd = data.groupby(['index']).apply(np.std).reset_index(drop=True).drop(['index'], axis=1)
        # dfmean = data.groupby(['index']).apply(np.mean).reset_index(drop=True)
        # dfstd = data.groupby(['index']).apply(np.std).reset_index(drop=True)
        
        fname = '%s_%s_%s_mean.pkl' %(strategy, transformation, ibase)
        with open(doc_save + '/' + fname, 'wb') as f:
            pickle.dump(dfmean, f)
        fname = '%s_%s_%s_std.pkl' %(strategy, transformation, ibase)
        with open(doc_save + '/' + fname, 'wb') as f:
            pickle.dump(dfstd, f)
        # check for 
        dfmean.columns
        # x = dfmean['Train Accuracy w']
        x = dfmean['Val Accuracy w and c']
        # choice == 'convergence'
        for i in range(1, len(x)):
            # z = x[:i]
            if equal_consecutive(x[:i], k=100, tol=1e-3):
                print(i)
                break
        k1 = i - 100 + 1
        # choice == 'valmax'
        k2 = np.argmax(x)
        
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 14))
        # plt.plot(x)
        # fig.savefig('SGD_iters.jpg', bbox_inches='tight')
        # dfmean.columns
        aa = dfmean.drop('n_features', axis=1).iloc[k1].tolist()
        bb = dfstd.drop('n_features', axis=1).iloc[k1].tolist()
        df_entry = pd.DataFrame([[clf] + [k1, nfeat, ibase] + aa + bb], columns=cnames )
        dt1 = dt1.append(df_entry)
        
        aa = dfmean.drop('n_features', axis=1).iloc[k2].tolist()
        bb = dfstd.drop('n_features', axis=1).iloc[k2].tolist()
        df_entry = pd.DataFrame([[clf] + [k2, nfeat, ibase] + aa + bb], columns=cnames )
        dt2 = dt2.append(df_entry)
        
    dt1 = dt1.sort_values('n_bases').reset_index(drop=True)
    fname = 'IRLDA_%s_%s_convergence.pkl' %(strategy, transformation)
    with open(doc_save + '/' + fname, 'wb') as f:
        pickle.dump(dt1, f)
    dt2 = dt2.sort_values('n_bases').reset_index(drop=True)
    fname = 'IRLDA_%s_%s_valmax.pkl' %(strategy, transformation)
    with open(doc_save + '/' + fname, 'wb') as f:
        pickle.dump(dt2, f)
    # log.columns
    # from matplotlib import pylab as plt
    
    # for cname in dfmean.columns:
    #     # x = dfmean[cname]
    #     dfmean = dfmean.sort_values('index').reset_index(drop=True)
    #     if cname not in ['Classifier', 'n_features', 'Time Used', 'index']:
    #         plt.plot(dfmean[cname])
    #         plt.legend()


def aggregate_all_log_result(doc_root="data/CNS", transformation="rff", rtype="valmax"):
    # doc_root = "data/CNS"
    # transformation='rff'
    # rtype = "convergence"; rtype="valmax"
    # doc_root = "data/CNS"; transformation='rff'; rtype = "convergence"
    gdcols = [
        'Classifier', 'n_iters', 'n_features', 'n_bases', 'Time Used Mean', 'Time Used Std', 
        'Train Accuracy w and c Mean',  'Train Accuracy w and c Std', 
        'Val Accuracy w and c Mean', 'Val Accuracy w and c Std']
    clfcols = [
        'Classifier', 'n_features', 'n_bases', 'Time Used Mean', 'Time Used Std', 
        'Train Accuracy Mean', 'Train Accuracy Std', 
        'Val Accuracy Mean', 'Val Accuracy Std']
    rescols = [
        'Classifier', 'n_iters', 'n_features', 'n_bases', 'Time Used Mean', 'Time Used Std', 
        'Train Accuracy Mean', 'Train Accuracy Std', 
        'Val Accuracy Mean', 'Val Accuracy Std']
    doc_name = doc_root + "/ResultSummary"
    filename = doc_name + "/Classifier_%s.pkl" %(transformation)
    with open(filename, 'rb') as f:
        dt = pickle.load(f)
    df_list = []
    for k in dt:
        tmp = dt[k]
        tmp = tmp.sort_values(by='n_bases').reset_index(drop=True)
        df = tmp[clfcols]
        df['n_iters'] = -1
        # df.columns
        df_list.append(df)
    # read the SGD and GD result.
    # filename = doc_name + "/IRLDA_SGD_%s_convergence.pkl" %(transformation)
    # filename = doc_name + "/IRLDA_SGD_%s_valmax.pkl" %(transformation)
    filename = doc_name + "/IRLDA_SGD_%s_%s.pkl" %(transformation, rtype)
    with open(filename, 'rb') as f:
        tmp = pickle.load(f)
    tmp = tmp.sort_values(by='n_bases').reset_index(drop=True)
    tmp.columns
    df = tmp[gdcols]
    df.columns = rescols
    df_list.append(df)
    
    # filename = doc_name + "/IRLDA_GD_%s_convergence.pkl" %(transformation)
    # filename = doc_name + "/IRLDA_GD_%s_valmax.pkl" %(transformation)
    filename = doc_name + "/IRLDA_GD_%s_%s.pkl" %(transformation, rtype)
    # fname = 'IRLDA_%s_%s_convergence.pkl' %(strategy, transformation)
    with open(filename, 'rb') as f:
        tmp = pickle.load(f)
    tmp = tmp.sort_values(by='n_bases').reset_index(drop=True)
    df = tmp[gdcols]
    df.columns = rescols
    df_list.append(df)
    
    # concate the data
    data = pd.concat(df_list)
    doc_save = doc_root + "/logall"
    if not os.path.exists(doc_save):
        print("NO this dir and make a new one.")
        os.makedirs(doc_save)
    
    filename = doc_save + '/logall_result_%s_%s.pkl' %(transformation, rtype)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    filename = doc_save + '/logall_result_%s_%s.csv' %(transformation, rtype)
    data.to_csv(filename, index=False, header=True)


def aggregate_main():
    # aggregate_all_log_result(doc_root="data/CNS", transformation='rff', rtype="valmax")
    doc_list = [
        "data/Leukemia", "data/Colon", "data/CNS", "data/Accent",
        "data/Audit", "data/ARCX", "data/ARCZ", "data/LSVT", "data/PGD"]
    # ## change doc_list if we have already had some documents aggregated.
    # doc_list = ["data/Accent"]
    # doc_list = ["data/ARCX", "data/ARCZ"]
    for doc_root in doc_list:
         # ["data/CNS", "data/Leukemia", "data/Colon"]
         strategy = 'Classifier'
         for transformation in ['rff', 'rbf']:
             classifier_result_summary(doc_root, strategy, transformation)
    
    for doc_root in doc_list:
         # ["data/CNS", "data/Leukemia", "data/Colon"]
         for strategy in ['GD', 'SGD']:
             # transformation = 'rff'
             for transformation in ['rff', 'rbf']:
                 IRLDA_result_summary(doc_root, strategy, transformation)
    
    for doc_root in doc_list:
        # transformation = 'rff'; 
        for transformation in ['rff', 'rbf']:
            for rtype in ['valmax', 'convergence']:
                aggregate_all_log_result(doc_root, transformation, rtype)


if __name__ == '__main__':
    print("YES!")
    # aggregate_main()


