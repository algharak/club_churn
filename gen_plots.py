from args_pg import parse_args
args = parse_args()
import numpy as np
from matplotlib import pyplot
from hyperopt import STATUS_OK,hp
import xgboost as xgb
from experiment import *
import hyperopt.pyll.base
from hyperopt.pyll.base import scope
from xgboost import XGBClassifier as xgb_kl
from sklearn.model_selection import KFold, train_test_split, GridSearchCV,validation_curve,learning_curve,StratifiedKFold,cross_val_score,StratifiedShuffleSplit
from sklearn.metrics import recall_score

def flatten_list (l):
    return [item for sublist in l for item in sublist]

def gen_plot(dict):
    tr_crv_title,te_crv_title = dict.keys()
    [crv_error] = dict[tr_crv_title].keys()
    [tr_crv_data] = dict[tr_crv_title].values() ; [te_crv_data] = dict[te_crv_title].values()
    x = np.arange(1,len(tr_crv_data)+1)
    fig, ax = pyplot.subplots(figsize=(8, 8))
    ax.plot(x,tr_crv_data, label='Train')
    ax.plot(x,te_crv_data, label='Test')
    ax.legend()
    pyplot.title(crv_error)
    pyplot.show()
    return

def objective(params,xt,yt,xe,ye):
    model = xgb_kl(**params)
    kfold = KFold(n_splits=4)
    cv_results = cross_val_score(model, xt, yt, cv=kfold,scoring='roc_auc')
    print("AUC: %.2f%% (%.2f%%)" % (cv_results.mean() * 100, cv_results.std() * 100))
    best_score = max(cv_results)
    loss = 1 - best_score
    #test_(model,xe,ye,**params)
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

def test_ (mod,x,y,**par):
    print('best params:   ', par)
    par['max_depth'] = int(par['max_depth'])
    par['n_estimators'] = int(par['n_estimators'])
    mod = xgb_kl(**par)
    best_mod = mod.fit(x,y, eval_metric='auc')
    ypred = best_mod.predict(x)
    rs = recall_score(y, ypred)
    print('The best tr recall score so far:  ', "{:.2%}".format(rs))
    args.plt_learn_cv = rs
    return


'''
space = {
    'objective': 'binary:logistic',
    #'scale_pos_weight': scope.txt(hp.choice('scale_pos_weight', [0.5,0.6])),
    #'booster': hp.choice('booster',['gbtree','dart']),
    'n_estimators':scope.int(hp.uniform('n_estimators', 8,128,3)),
    'max_depth': scope.int(hp.uniform('max_depth', 2,32,2)),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-3), np.log(0.99),
    ‘gamma': hp.loguniform('gamma', np.log(1e-6), np.log(64)),
    'min_child_weight': hp.loguniform('min_child_weight',np.log(1e-6), np.log(32)),
    
'subsample': hp.uniform('subsample', 0.5,1.0),

    'colsample_bytree': hp.uniform('colsample_bytree’,0.3,1),

    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-6), np.log(2)),

    'reg_lambda': hp.loguniform('reg_lambda',np.log(1e-6), np.log(2),
}


'''

space = {
    'objective': 'binary:logistic',
    #'scale_pos_weight': scope.txt(hp.choice('scale_pos_weight', [0.5,0.6])),
    #'booster': hp.choice('booster',['gbtree','dart']),
    'n_estimators': scope.int(hp.quniform('n_estimators', 20, 128,3)),
    'max_depth': scope.int(hp.quniform('max_depth', 2, 32, 2)),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-3), np.log(0.99)),
    'gamma': hp.loguniform('gamma', np.log(1e-1), np.log(64)),
    'min_child_weight': hp.loguniform('min_child_weight', np.log(1e-2), np.log(32)),
    'subsample': hp.uniform('subsample', 0.5, .99),
    'colsample_bytree': hp.uniform('colsample_bytree',0.3,1),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-2), np.log(4)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-2), np.log(4))}

'''
space = {
    'objective': 'binary:logistic',
    #'scale_pos_weight': scope.txt(hp.choice('scale_pos_weight', [0.5,0.6])),
    #'booster': hp.choice('booster',['gbtree','dart']),
    'n_estimators':scope.int(hp.quniform('n_estimators', 2,35,2)),
    'max_depth': scope.int(hp.quniform('max_depth', 2,40,2)),
    'learning_rate': hp.uniform('learning_rate', 1e-3, 0.9),
    'gamma': hp.uniform('gamma', 1e-1, 50),
    'min_child_weight': hp.uniform('min_child_weight',1e-3 ,32),
    'subsample': hp.uniform('subsample', 0.5,1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.001, 0.999),
    'reg_alpha': hp.uniform('reg_alpha', 1e-1, 8.0),
    'reg_lambda': hp.uniform('reg_lambda',1e-1, 8.0),
}

space_ = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'booster': hp.choice('booster',['gbtree','dart']),
    'max_depth': scope.int(hp.quniform('max_depth', 2,100,2)),
    'eta': hp.uniform('eta', 1e-4, 0.9),
    'gamma': hp.uniform('gamma', 1e-4, 32),
    'min_child_weight': hp.uniform('min_child_weight',1e-6 ,32),
    'subsample': hp.uniform('subsample', 0.5,1),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0, 0.999),
    'alpha': hp.uniform('alpha', 1e-4, 4.0),
    'lambda': hp.uniform('lambda',1e-4, 4.0),
}
'''