
import pandas as pd
import numpy as np
from args_pg import *
args = parse_args()
from xgboost import XGBClassifier as xgb_kl
from sklearn.model_selection import cross_val_score,StratifiedShuffleSplit
from ax import optimize

baseparam=dict(objective='binary:logistic')
baseparam.update(dict(n_estimators=800))
baseparam.update(dict(learning_rate=0.1))
baseparam.update(dict(scale_pos_weight=1))
baseparam.update(dict(booster='dart'))
#phase 1-wide focus
#baseparam.update(dict(max_depth=6,8))
#baseparam.update(dict(min_child_weight=1,7))
baseparam.update(dict(gamma=0))
baseparam.update(dict(subsample=0.8))
baseparam.update(dict(colsample_bytree=0.8))

Ax_n_trials  = 20
Ax_max_iter  = 10
Ax_par=[{"name": "max_depth",           "type": "choice",        "values": [6,7],   "value_type": "int"},
        {"name": "min_child_weight",    "type": "choice",        "values": [1,2],   "value_type": "int"}]
colnames = ['objective']+[item['name'] for item in Ax_par]

def tune_params(sc):
    print(('***********     Begin Grid Search for HPs'))
    myobj= obj_wrapper(sc)
    result_pd = pd.DataFrame([],index=np.arange(Ax_n_trials),columns= colnames,dtype=int)
    for iter in range(Ax_n_trials):
        best_parameters, best_values, _,_= optimize(Ax_par,evaluation_function=myobj.ax_optim,minimize=True,total_trials=Ax_max_iter)
        score = best_values[0]
        result_pd.iloc[iter,:] = {**score,**best_parameters}
        print ('Best Parameters:  ',best_parameters)
        print ('Best Values:  ',best_values)
    print(result_pd.head())
    result_pd[result_pd.objective == result_pd.objective.min()]
    full_param = {**result_pd.iloc[0,:].to_dict(), **baseparam}
    full_param = adjust_dtype(full_param)
    print('full_params are:  ',full_param)
    return full_param, best_values

class obj_wrapper ():
    def __init__(self,sc):
        self.xtr = sc.xtr
        self.xte = sc.xte
        self.ytr = sc.ytr
        self.yte = sc.yte
    def ax_optim(self,par):
        full_param = {**par, **baseparam}
        mod = xgb_kl(**full_param)
        kfold = StratifiedShuffleSplit(n_splits=4)
        cv_results = cross_val_score(mod, self.xtr, self.ytr, cv=kfold, scoring='recall')
        loss = 1 - max(cv_results)
        return loss

def adjust_dtype(di):
    intcols = ["max_depth",'min_child_weight']
    for label in intcols:
        di[label] = int(di[label])
    return di