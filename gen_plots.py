from args_pg import *
args = parse_args()
import numpy as np
from matplotlib import pyplot
from hyperopt import STATUS_OK,hp
from experiment import *
import hyperopt.pyll.base
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
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
    model = xgb_kl(**full_param)
    kfold = StratifiedShuffleSplit(n_splits=4)
    cv_results = cross_val_score(model, xt, yt, cv=kfold,scoring='recall')
    print("Recall: %.2f%% (%.2f%%)" % (cv_results.mean() * 100, cv_results.std() * 100))
    best_score = max(cv_results)
    loss = 1 - best_score
    aaa=params.values()
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


#space={'max_depth': scope.int(hp.quniform('max_depth',2,8,1))} #produces float
space={'max_depth': hp.choice('max_depth',[3,4,5])}
space.update({'min_child_weight': hp.choice('min_child_weight',[1,2,3])})
#space.update({'gamma': scope.int(hp.choice('gamma',np.arange(1,5,dtype=int)))})
#space.update({'max_depth': hp.quniform('max_depth',3,8,1)})
#space.update({'max_depth': scope.int(hp.quniform('max_depth',3,8,1))})
#space.update({'max_depth': sample(scope.int(hp.uniform('max_depth',3,8)))})
#space.update({'max_depth': hp.choice('max_depth',np.arange(3,8,dtype=int))})
#space.update({'reg_alpha': hp.loguniform('reg_alpha',-0.9,-0.8)})
#space={'reg_alpha': hp.quniform('reg_alpha',0.3,0.35,q=.01)}
#space.update({'colsample_bytree': hp.quniform('colsample_bytree',0.6,1.0,q=.1)})
#space={'min_child_weight': scope.int(hp.quniform('min_child_weight',2,7,q=1))}
#space={'min_child_weight': hp.choice('min_child_weight',[11,2,3,1,4])}


#guniform('min_child_weight',np.log(1e-6), np.log(32)),'subsample': hp


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
