#!/usr/bin/env python3
import os
import pickle
import pandas as pd
import itertools


def best_result(dname, transformation, rtype):
    # dname = "CNS"
    # transformation = "rff"
    # rtype = "valmax"
    doc_name = "data/" + dname + "/logall"
    filename = doc_name + '/logall_result_%s_%s.pkl' %(transformation, rtype)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    data.columns
    
    clf_list = list(set(data['Classifier']))
    df_nfeat = pd.DataFrame(columns=clf_list)
    df_acc = pd.DataFrame(columns=clf_list)
    df_std = pd.DataFrame(columns=clf_list)
    df_niter = pd.DataFrame(columns=clf_list)
    
    nfeat, acc, std, niter = [], [], [], []
    for clf in clf_list:
        dt = data[data['Classifier'] == clf]
        dt = dt.sort_values(
            by=['Val Accuracy Mean', 'Val Accuracy Std'], ascending=[False, True], 
            axis=0, inplace=False)
        dt = dt.reset_index(drop=True)
        # nfeat.append(dt.loc[0, 'n_features'])  # n_features
        nfeat.append(dt.loc[0, 'n_bases'])  # n_bases
        acc.append(dt.loc[0, 'Val Accuracy Mean'])
        std.append(dt.loc[0, 'Val Accuracy Std'])
        niter.append(dt.loc[0, 'n_iters'])
    
    df_entry = pd.DataFrame([acc], columns=clf_list )
    df_acc = df_acc.append(df_entry)
    
    df_entry = pd.DataFrame([std], columns=clf_list )
    df_std = df_std.append(df_entry)
    
    df_entry = pd.DataFrame([nfeat], columns=clf_list )
    df_nfeat = df_nfeat.append(df_entry)
    
    df_entry = pd.DataFrame([niter], columns=clf_list )
    df_niter = df_niter.append(df_entry)
    
    return df_acc, df_std, df_nfeat, df_niter


def best_main():
    somelists = [
        ["Colon", "CNS", "Leukemia", "Accent", "Audit", "ARCX", "ARCZ", "LSVT", "PGD"],
        ["rff", "rbf"], 
        ["convergence", "valmax"]
    ]
    
    inames = []
    data_acc = pd.DataFrame()
    data_std = pd.DataFrame()
    data_nfeat = pd.DataFrame()
    data_niter = pd.DataFrame()
    
    for element in itertools.product(*somelists):
        print(element)
        inames.append('_'.join(element))
        dname, transformation, rtype = element
        df_acc, df_std, df_nfeat, df_niter = best_result(dname, transformation, rtype)
        data_acc = data_acc.append(df_acc)
        data_std = data_std.append(df_std)
        data_nfeat = data_nfeat.append(df_nfeat)
        data_niter = data_niter.append(df_niter)
    
    data_acc.insert(loc=0, column='inames', value=inames)
    data_acc.reset_index(drop=True, inplace=True)
    data_std.insert(loc=0, column='inames', value=inames)
    data_std.reset_index(drop=True, inplace=True)
    data_nfeat.insert(loc=0, column='inames', value=inames)
    data_nfeat.reset_index(drop=True, inplace=True)
    data_niter.insert(loc=0, column='inames', value=inames)
    data_niter.reset_index(drop=True, inplace=True)
    
    doc_save = "BestResults"
    if not os.path.exists(doc_save):
        print("NO this dir and make a new one.")
        os.makedirs(doc_save)
    
    filename = doc_save + '/best_result_acc.csv'
    data_acc.to_csv(filename, index=False, header=True)
    filename = doc_save + '/best_result_std.csv'
    data_std.to_csv(filename, index=False, header=True)
    filename = doc_save + '/best_result_nfeat.csv'
    data_nfeat.to_csv(filename, index=False, header=True)
    filename = doc_save + '/best_result_niter.csv'
    data_niter.to_csv(filename, index=False, header=True)
    
    algorithms = [
        'inames',
        'AdaBoostClassifier',
        'GradientBoostingClassifier',
        'OLS_GD',
        'OLS_batchSGD']
    
    filename = doc_save + '/best_result_acc_short.csv'
    df = data_acc[algorithms]
    df.to_csv(filename, index=False, header=True)
    filename = doc_save + '/best_result_std_short.csv'
    df = data_std[algorithms]
    df.to_csv(filename, index=False, header=True)
    filename = doc_save + '/best_result_nfeat_short.csv'
    df = data_nfeat[algorithms]
    df.to_csv(filename, index=False, header=True)
    filename = doc_save + '/best_result_niter_short.csv'
    df = data_niter[algorithms]
    df.to_csv(filename, index=False, header=True)
    return data_acc, data_std, data_nfeat, data_niter


if __name__ == '__main__':
    print("YES!")
    data_acc, data_std, data_nfeat, data_niter = best_main()
    algorithms = [
        'inames',
        'AdaBoostClassifier',
        'GradientBoostingClassifier',
        'OLS_GD',
        'OLS_batchSGD']
    
    dt_acc = data_acc[algorithms]
    
    
    
    
    


