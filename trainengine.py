from args_pg import parse_args
args = parse_args()
import pandas as pd
from sklearn.utils  import shuffle
from transform import transform_,procss
import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV,validation_curve,learning_curve,StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix, mean_squared_error,accuracy_score,classification_report
from gen_plots import gen_plot
from xgboost import XGBClassifier as xgb_kl
import matplotlib.pylab as plt




def get_master_pd(path):
    return pd.read_csv(path)
from utils import pdcol2np


def exp_shuf_fold(fr):
    newfrm = fr.copy()
    for iter in range(args.shuffle):
        newfrm = shuffle(newfrm)
    te_tr_list = []
    if args.Kfolds > 1:
        kf = KFold(n_splits=args.Kfolds)
        for train, test in kf.split(newfrm):
            tr_epochs = args.epochs*[newfrm.iloc[train]]
            tr_epochs = pd.concat(tr_epochs)
            tr_epochs = shuffle(tr_epochs)
            te_tr_list.append([tr_epochs]+[newfrm.iloc[test]])
    if not te_tr_list:
        te_tr_list.append(newfrm)
        return te_tr_list

def extract_cmds (e):
    do_train = eval(eval('args.'+e)['do_train'])
    do_train = eval(eval('args.'+e)['do_train'])
    params =eval('args.'+e)['param']
    rounds =eval('args.'+e)['rounds']
    get_dataset =eval(eval('args.'+e)['get_dataset'])
    do_cv = eval(eval('args.'+e)['do_cv'])
    return do_train,do_cv,rounds,params, get_dataset


def train_ (trte_li,y_shuff = True):
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

    do_test1=False
    if do_test1:
        param_test1 = {
            'max_depth': range(3, 5 , 1),
            'min_child_weight': range(1, 3, 1)}
        gsearch1 = GridSearchCV(estimator=xgb_kl(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                        objective='binary:logistic', nthread=1, scale_pos_weight=1,
                                                        seed=27,refit=True),
                                param_grid=param_test1, scoring='roc_auc', n_jobs=4, cv=5)
        gsearch1.fit(xtr,pdcol2np(ytr))
        gsearch1.best_params_, gsearch1.best_score_

    do_test3=False
    if do_test3:
        param_test3 = {'gamma': [i / 1000.0 for i in range(0, 300,30)]}
        gsearch3 = GridSearchCV(estimator=xgb_kl(learning_rate=0.1, n_estimators=500, max_depth=4,
                                                 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                 objective='binary:logistic', scale_pos_weight=1,
                                                 seed=32, refit=True),
                                param_grid=param_test3, scoring='roc_auc', cv=5)
        gsearch3.fit(xtr, pdcol2np(ytr))
        gsearch3.best_params_, gsearch3.best_score_
        print()

    do_test4 =  False
    if do_test4:
        print('******** tunning subsample and colsample_bytree')
        param_test4 = {
            'subsample': [i / 10.0 for i in range(6, 10)],
            'colsample_bytree': [i / 10.0 for i in range(6, 10)]}
        gsearch4 = GridSearchCV(estimator=xgb_kl(learning_rate=0.1, n_estimators=500, max_depth=4,
                                                        min_child_weight=.001, gamma=0.03, subsample=0.8, colsample_bytree=0.8,
                                                        objective='binary:logistic', scale_pos_weight=1,
                                                        seed=27,refit=True),
                                param_grid=param_test4, scoring='roc_auc', cv=5)
        gsearch4.fit(xtr, pdcol2np(ytr))
        gsearch4.best_params_, gsearch4.best_score_

    do_test5 = False
    if do_test5:
        print('********  fine tunning subsample and colsample_bytree ')
        param_test5 = {
            'subsample': [i / 100.0 for i in range(60, 80,5)],
            'colsample_bytree': [i / 100.0 for i in range(70, 95, 5)]}
        gsearch5 = GridSearchCV(estimator=xgb_kl(learning_rate=0.1, n_estimators=500, max_depth=4,
                                                 min_child_weight=1, gamma=0.03, subsample=0.6, colsample_bytree=0.8,
                                                 objective='binary:logistic', scale_pos_weight=1,
                                                 seed=27, refit=True),
                                param_grid=param_test5, scoring='roc_auc', cv=10)
        gsearch5.fit(xtr, pdcol2np(ytr))
        gsearch5.best_params_, gsearch5.best_score_

    do_test6 = False
    if do_test6:
        print('******** tunning reg_alpha ')
        param_test6 = {
            'reg_alpha': [1, 2,4,8,10,20]}
        gsearch6 = GridSearchCV(estimator=xgb_kl(learning_rate=0.1, n_estimators=500, max_depth=4,
                                                 min_child_weight=1, gamma=0.03, subsample=0.6, colsample_bytree=0.8,
                                                 objective='binary:logistic', scale_pos_weight=1,
                                                 seed=27, refit=True),
                                param_grid=param_test6, scoring='roc_auc', cv=5)
        gsearch6.fit(xtr, pdcol2np(ytr))
        gsearch6.best_params_, gsearch6.best_score_

    do_test7 = False
    if do_test7:
        param_test7 = {
            'learning_rate': [0.02, 0.03, 0.04, 0.05, 0.06, 0.12],'reg_alpha':[10]}
        print('******** tunning LR ')
        gsearch7 = GridSearchCV(estimator=xgb_kl(learning_rate=0.1, n_estimators=1000, max_depth=4,
                                                 min_child_weight=1, gamma=0.01, subsample=0.6, colsample_bytree=0.8,
                                                 objective='binary:logistic', scale_pos_weight=1,
                                                 seed=27, refit=True),param_grid=param_test7, scoring='roc_auc', cv=5)
        gsearch7.fit(xtr, pdcol2np(ytr))
        gsearch7.best_params_, gsearch7.best_score_
    do_plot = True
    if do_plot:
        final_model = xgb_kl(learning_rate=0.1, n_estimators=1000, max_depth=8,
                                                 min_child_weight=1, gamma=0.03, subsample=0.6, colsample_bytree=0.8,
                                                 objective='binary:logistic',seed=27,reg_alpha=10
                             ,reg_lambda=0,scale_pos_weight=0.41)
        bst = final_model.fit(xtr, ytr, eval_set=eval_set,eval_metric='auc')
        eval_result = bst.evals_result()
        ypred = bst.predict(xte)
        print(confusion_matrix(yte, ypred))
        print(classification_report(yte, ypred))
        print('The Accuracy is:  ', "{:.2%}".format(accuracy_score(yte, ypred)))
        gen_plot(eval_result)
    exit()

def prep_data (fr,pd=True,y_shuff=False,standardize=True):
    print ('\n***********    Retrieving Train/Test Data')
    labels = fr.columns[-1]
    y = procss(fr[[labels]])
    x = procss(fr.drop(labels, axis=1))
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=args.trte_split, random_state=42)
    if y_shuff:
        ytr = shuffle(ytr)
    if not pd:
        ytr = pdcol2np(ytr) ; yte = pdcol2np(yte)
        xtr = pdcol2np(xtr) ;   xte = pdcol2np(xte)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=args.trte_split, random_state=42)
    return [xtr, xte, ytr, yte]

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
    feat_imp=pd.Series(alg.get_booster().get_score(importance_type='weight')).sort_values(ascending=False)
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    return

def extract_ds():
    pd_master = get_master_pd(args.src_file)
    trte_lst = exp_shuf_fold (pd_master)
    return trte_lst

def main():
    trte_pd= extract_ds()
    print ('***********     Dataset loading was successful')
    for frame in trte_pd:
        trte_lst = prep_data(frame)
        train_ (trte_lst)
if __name__ == "__main__":
    main()
