
import pandas as pd
import numpy as np
from args_pg import *
args = parse_args()
from xgboost import XGBClassifier as xgb_kl
from sklearn.model_selection import cross_val_score,StratifiedShuffleSplit
from ax import optimize

baseparam=dict(objective='binary:logistic')
baseparam.update(dict(n_estimators=1000))
baseparam.update(dict(learning_rate=0.1))
baseparam.update(dict(scale_pos_weight=1))
baseparam.update(dict(booster='dart'))
baseparam.update(dict(max_depth=3))
baseparam.update(dict(min_child_weight=2))
baseparam.update(dict(gamma=0.05))
baseparam.update(dict(subsample=0.8))
baseparam.update(dict(colsample_bytree=0.6))
baseparam.update(dict(reg_alpha=0.01))

Ax_n_trials  = 5
Ax_max_iter  = 30
#Ax_par=[{"name": "reg_alpha","type": "range","bounds": [1e-3,10],#"value_type": "float",'log_scale':True}]

colnames = []
#colnames = ['objective']+[item['name'] for item in Ax_par]



def tune_params(sc):
    print(('***********     Begin Grid Search for HPs'))
    myobj= obj_wrapper(sc)
    result_pd = pd.DataFrame([],index=np.arange(Ax_n_trials),columns= colnames,dtype=int)
    for iter in range(Ax_n_trials):
        best_parameters, best_values, _,_= optimize(Ax_par,evaluation_function=myobj.ax_optim,minimize=True,total_trials=Ax_max_iter,)
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

'''

Ax_par=[{"name": "max_depth",           "type": "choice",        "values": [6,7,8],   "value_type": "int"},
        {"name": "min_child_weight",    "type": "choice",        "values": [1,2,3],   "value_type": "int"}]
====================================================================================================================
Ax_par=[{"name": "gamma","type": "range","bounds": [1e-2,0.5],"value_type": "float",'log_scale':True}]
note the values were 0.379,0.235,0.05,0.052,0.04 picked 0.05

====================================================================================================================
Ax_par=[{"name": "subsample",           "type": "choice",        "values": [0.6,0.7,0.8],   "value_type": "float"},
        {"name": "colsample_bytree",    "type": "choice",        "values": [0.6,0.7,0.8],   "value_type": "float"}]

objective  subsample  colsample_bytree
0   0.444444        0.8               0.8
1   0.440476        0.8               0.8
2   0.412698        0.8               0.6
3   0.480159        0.8               0.7
4   0.464286        0.6               0.8
======================================================
Ax_par=[{"name": "reg_alpha","type": "range","bounds": [1e-4,10],"value_type": "float",'log_scale':True}]
reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
'''