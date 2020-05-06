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
