from args_pg import parse_args
args = parse_args()
import pandas as pd
from sklearn.utils  import shuffle
from transform import transform_,procss
from utils import myshuffle
import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV,validation_curve,learning_curve,StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix, balanced_accuracy_score,accuracy_score,classification_report
from gen_plots import gen_plot
from xgboost import XGBClassifier as xgb_kl
import matplotlib.pylab as plt
import random
from experiment import *

def get_master_pd(path):
    return pd.read_csv(path)
from utils import pdcol2np

def extract_cmds (e):
    do_train = eval(eval('args.'+e)['do_train'])
    do_train = eval(eval('args.'+e)['do_train'])
    params =eval('args.'+e)['param']
    rounds =eval('args.'+e)['rounds']
    get_dataset =eval(eval('args.'+e)['get_dataset'])
    do_cv = eval(eval('args.'+e)['do_cv'])
    return do_train,do_cv,rounds,params, get_dataset

def modelfit(alg, x,y, xte,yte, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        y=pdcol2np(y)
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x, label=y)
        remove = alg.get_params()['n_estimators']
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    # Fit the algorithm on the data
    #alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')
    alg.fit(x, y, eval_metric='auc')
    # Predict training set:
    dtrain_predictions = alg.predict(x)
    dtest_prediction = alg.predict(xte)
    dtrain_predprob = alg.predict_proba(x)[:, 1]
    dtest_predprob = alg.predict_proba(xte)[:, 1]
    # Print model report:
    print("\n********** Model Report **********")
    print("Train Accuracy : %.4g" % metrics.accuracy_score(y, dtrain_predictions))
    print("Test Accuracy : %.4g" % metrics.accuracy_score(yte, dtest_prediction))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))
    print("AUC Score (Test): %f" % metrics.roc_auc_score(yte, dtest_predprob))
    #feat_imp=pd.Series(alg.get_booster().get_score(importance_type='weight')).sort_values(ascending=False)
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')
    return

def train_ (trte_li,y_shuff = False):
    print ('***********     Start The Training Process')
    [xtr, xte, ytr, yte] = trte_li
    predictors = [x for x in xtr.columns]
    eval_set = [(xtr,ytr),(xte,yte)]
    print(('***********     Pick LR & nEstimators'))
    xgb1 = xgb_kl(
        learning_rate=0.05,
        n_estimators=200,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=0.41,
        seed=27)
    modelfit(xgb1, xtr,ytr,xte,yte,predictors)
    exit()

    do_plot = True
    if do_plot:
        final_model = xgb_kl(learning_rate=0.05, n_estimators=25000, max_depth=5,
                                                 min_child_weight=1, gamma=3, subsample=0.6, colsample_bytree=0.6,
                                                 objective='binary:logistic',seed=27,reg_alpha=38
                             ,reg_lambda=38,scale_pos_weight=0.41)
        if y_shuff:
            ytr = myshuffle(ytr)
        ytr = pdcol2np(ytr)
        bst = final_model.fit(xtr, ytr, eval_set=eval_set,eval_metric='auc')
        eval_result = bst.evals_result()
        ypred = bst.predict(xte)
        print(confusion_matrix(yte, ypred))
        print(classification_report(yte, ypred))
        print('The Accuracy is:  ', "{:.2%}".format(accuracy_score(yte, ypred)))
        print('The Balanced Accuracy is:  ', "{:.2%}".format(balanced_accuracy_score(yte, ypred)))
        gen_plot(eval_result)
    exit()

def extract_ds():
    pd_master = get_master_pd(args.src_file)
    if args.shuffle:
        pd_master = myshuffle(pd_master)
    print ('***********     Dataset loading was successful')
    return pd_master

def gen_dummy(fr):
    nrows = fr.shape[0]
    colname = fr.columns[-1]
    ratio = 0.0
    lst = random.sample (range(nrows),round(nrows*ratio))
    col = fr.loc[:,colname].values.copy()
    for i in range(nrows):
        col[i] = True if col[i] == 'CANCELLED' else False
        col[i] = not col[i] if i in lst else col[i]
    newfrm = pd.DataFrame(col,columns=['Dummy'])
    temp = pd.concat([newfrm,fr],axis=1)
    return temp

def main():
    trte_pd= extract_ds()
    run_plan = dset (trte_pd)
    xtrxtr = run_plan.xtr()
    xtexte = run_plan.xte()
    ytrytr = run_plan.ytr()
    yteyte = run_plan.yte()
    print ()
    train_ (run_plan)
if __name__ == "__main__":
    main()
