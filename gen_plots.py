from args_pg import parse_args
args = parse_args()
import numpy as np
from matplotlib import pyplot
from hyperopt import STATUS_OK,hp
from experiment import *
import hyperopt.pyll.base
from hyperopt.pyll.base import scope
from xgboost import XGBClassifier as xgb_kl
from sklearn.model_selection import KFold, train_test_split, GridSearchCV,validation_curve,learning_curve,StratifiedKFold,cross_val_score,StratifiedShuffleSplit
from sklearn.metrics import recall_score

def flatten_list (l):
    return [item for sublist in l for item in sublist]

def gen_plot(dict):
    tr_crv_dict,te_crv_dict = dict.items()
    tr_crv  = tr_crv_dict[1]
    tr_crv_dt  = 1-np.array(tr_crv['recall'])
    te_crv = te_crv_dict[1]
    te_crv_dt = 1- np.array(te_crv['recall'])
    x = np.arange(1,len(te_crv_dt)+1)
    fig, ax = pyplot.subplots(figsize=(8, 8))
    bottom, top = pyplot.ylim()
    pyplot.ylim (top = 0.9)
    pyplot.ylim (bottom = 0.2)
    pyplot.grid(b=True, which='both', axis='both')
    ax.plot(x,tr_crv_dt, label='Train')
    ax.plot(x,te_crv_dt, label='Test')
    ax.legend()
    pyplot.title('Train vs. Test Recall')
    pyplot.show()
    return

def objective(params,xt,yt,xe,ye):
    full_param = {**params,**args.base_param}
    #params.update(args.base_param)

    model = xgb_kl(**full_param)
    #kfold = KFold(n_splits=4)
    #kfold = StratifiedKFold(n_splits=4)
    kfold = StratifiedShuffleSplit(n_splits=4)
    cv_results = cross_val_score(model, xt, yt, cv=kfold,scoring='recall')
    print("Recall: %.2f%% (%.2f%%)" % (cv_results.mean() * 100, cv_results.std() * 100))
    best_score = max(cv_results)
    loss = 1 - best_score
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
    return



#space={'min_child_weight': hp.choice('min_child_weight',[5,9])}
space={'min_child_weight': scope.int(hp.quniform('min_child_weight',1,5,q=1))}
#space={'min_child_weight': hp.choice('min_child_weight',[11,2,3,1,4])}




'''
scope.int(hp.quiniform('my_param', 1, 100, q=1))
#'scale_pos_weight': scope.txt(hp.choice('scale_pos_weight', [0.5,0.6])),
    #'booster': hp.choice('booster',['gbtree','dart']),
    # 'n_estimators':scope.int(hp.uniform('n_estimators', 8,128,3)),
    'max_depth': scope.int(hp.uniform('max_depth', 2,32,2)),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-3), np.log(0.99)),'gamma': hp.loguniform('gamma', np.log(1e-6), np.log(64)),'min_child_weight': hp.loguniform('min_child_weight',np.log(1e-6), np.log(32)),'subsample': hp.uniform('subsample', 0.5,1.0),'colsample_bytree': hp.uniform('colsample_bytree’,0.3,1),'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-6), np.log(2)),'reg_lambda': hp.loguniform('reg_lambda',np.log(1e-6), np.log(2))
}


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

'''