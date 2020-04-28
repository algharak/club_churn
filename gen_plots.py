from args_pg import parse_args
args = parse_args()
import numpy as np
from matplotlib import pyplot
from hyperopt import STATUS_OK,hp
import xgboost as xgb
from experiment import *
import hyperopt.pyll.base
from hyperopt.pyll.base import scope

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

def objective(params,xt,yt):
    tr_set = xgb.DMatrix(xt,label=yt)
    cv_results = xgb.cv(params, tr_set, nfold=6, num_boost_round=100,
                        early_stopping_rounds=10, metrics='rmse', seed=50)
    best_score = max(cv_results['test-rmse-mean'])
    loss = 1 - best_score
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

space = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'booster': hp.choice('booster',['gbtree','dart']),
    'max_depth': scope.int(hp.quniform('max_depth', 2,100,2)),
    'eta': hp.uniform('eta', 1e-4, 0.9),
    'gamma': hp.uniform('gamma', 1e-4, 32),
    'min_child_weight': hp.uniform('min_child_weight',1e-6 ,32),

    'subsample': hp.uniform('subsample', 0.5,1),
    'colsample_bytree': hp.loguniform('colsample_by_tree', -10, 0),
    'alpha': hp.uniform('alpha', 1e-6, 2.0),
    'lambda': hp.uniform('lambda',1e-6, 2.0),

}