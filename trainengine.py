import argparse
import pandas as pd
from sklearn.utils  import shuffle
from transform import transform_
import pickle
#import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV,validation_curve
from sklearn.ensemble import GradientBoostingClassifier as xgb

from sklearn.metrics import confusion_matrix, mean_squared_error,accuracy_score,classification_report
import os

####setup parameters
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_ds", type=str, default='./dataset/club_churn_source.csv')
    #parser.add_argument("--list", type=int, default=[2,2])
    #parser.add_argument("--trtesplit", type=float, default=0.1)
    parser.add_argument("--shuffle", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--Kfolds", type=int, default=10)
    parser.add_argument("--xgb_param", type=dict, default={'max_depth': 12, 'eta': 0.025,'n_estimators': 45})
    args = parser.parse_args()
    return args

def load_datasets(trte_lst):
    X_tr = []
    Y_tr = []
    X_te = []
    Y_te = []
    flat_list = [item for sublist in trte_lst for item in sublist]
    trlst, telst = flat_list[::2],flat_list[1::2]
    for i in range(len(trlst)):
        Y_tr.append(trlst[i].iloc[:,0].values.astype(int))
        X_tr.append(trlst[i].iloc[:,1:].values.astype(float))
        Y_te.append(telst[i].iloc[:, 0].values.astype(int))
        X_te.append(telst[i].iloc[:, 1:].values.astype(float))
    return zip(X_tr,Y_tr,X_te,Y_te)

def get_master_pd(path):
    return pd.read_csv(path)

def exp_shuf_fold(fr,arg):
    newfrm = fr.copy()
    if arg.epochs > 1:
        for i in range(arg.epochs-1):
            newfrm = newfrm.append(fr)
    for iter in range(arg.shuffle):
        newfrm = shuffle(newfrm)
    kf = KFold(n_splits=arg.Kfolds)
    te_tr_list = []
    for train, test in kf.split(newfrm):
        te_tr_list.append([newfrm.iloc[train]]+[newfrm.iloc[test]])
    return te_tr_list

def train_evaluate (tr_ev_dt):
    args = parse_args()
    for Xtr,Ytr,Xte,Yte in tr_ev_dt:
        print(('***********     Begin Training'))
        xgb_model = xgb.XGBClassifier(**args.xgb_param).fit(Xtr, Ytr)
        #xgb_model = xgb.XGBClassifier(n_estimators=45, learning_rate=0.025,max_depth=12, random_state=0).fit(Xtr, Ytr)
        Y_pred = xgb_model.predict(Xte)
        print(('***********     Begin Evaluation'))
        print('   mse      is :  ', mean_squared_error(Yte, Y_pred))
        print('   accuracy is :  ', accuracy_score(Yte, Y_pred))
        print('************  report  ************  ')
        print(classification_report(Yte, Y_pred))
        train_scores, valid_scores = validation_curve(xgb.XGBClassifier, Xtr, Ytr, "alpha",np.logspace(-7, 3, 3),cv = 5)
        print('hello')
        '''
        clf = GridSearchCV(xgb_model,
                           {'max_depth': [8,12],
                            'n_estimators': [35,45],
                            'eta': [0.025]}, verbose=0)
        clf.fit(Xtr, Ytr)
        print(clf.best_score_)
        print(clf.best_params_)

        '''

def extract_ds(arg):
    pd_master = get_master_pd(arg.source_ds)
    trte_lst = exp_shuf_fold (pd_master,arg)
    return trte_lst

def main():
    args = parse_args()
    trte_lst= extract_ds(args)
    trte_proc_lst = transform_ (trte_lst)
    print ('***********    Dataset Preprocessing was successful')
    tr_eval_data = load_datasets(trte_proc_lst)
    print ('***********     Dataset loading was successful')
    train_evaluate (tr_eval_data)




if __name__ == "__main__":
    main()
