
from args_pg import *
args = parse_args()
from xgboost import XGBClassifier as xgb_kl
from sklearn.model_selection import cross_val_score,StratifiedShuffleSplit
from ax import optimize

baseparam=dict(objective='binary:logistic')
baseparam.update(dict(n_estimators=700))
baseparam.update(dict(learning_rate=0.1))
baseparam.update(dict(scale_pos_weight=1))
baseparam.update(dict(booster='dart'))
#baseparam.update(dict(max_depth=4))
#baseparam.update(dict(min_child_weight=5))
baseparam.update(dict(gamma=0))
baseparam.update(dict(subsample=0.8))
baseparam.update(dict(colsample_bytree=0.8))

Ax_max_iter  =  3
Ax_par=[{"name": "max_depth",           "type": "choice",        "values": [2,4,6,8,10],},
        {"name": "min_child_weight",    "type": "choice",        "values": [1,3,5,7],},]

def tune_params(sc):
    print(('***********     Begin Grid Search for HPs'))
    myobj= obj_wrapper(sc)
    best_parameters, best_values, _,_  = optimize(Ax_par,evaluation_function=myobj.ax_optim,minimize=True,total_trials=Ax_max_iter)
    print ('Best Parameters:  ',best_parameters)
    full_param = {**best_parameters, **baseparam}
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
