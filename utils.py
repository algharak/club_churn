from args_pg import parse_args
args = parse_args()
import pandas as pd
import numpy as np
from sklearn.utils  import shuffle


def pdcol2np (col):
    nrows = col.shape[0]
    ncols = col.shape[1]
    colnp = col.values.reshape((nrows,ncols))
    if ncols==1:
        colnp = col.values.reshape((nrows))
    return colnp

def myshuffle (x):
    for iter in range(args.shuffle):
        x=shuffle(x)
    x = x.reset_index(drop=True)
    return x
'''



def train_ (trte_pd):
    print('\n Hello There')
    labels = trte_pd.columns[-1]
    y = pdcol2np(procss(trte_pd [[labels]]))
    x = pdcol2np(procss(trte_pd.drop(labels, axis=1)))
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=args.trte_split, random_state=42)
    #np.random.shuffle(ytr)
    eval_set = [(xtr,ytr),(xte,yte)]
    print(('***********     Begin Training'))
    model = xgb(**args.xgb['param'])
    bst = model.fit(xtr,ytr,eval_set=eval_set)
    #bst = xgb(**args.xgb['param']).fit(xtr,ytr)
    #bst = xgb(**args.xgb['param']).fit(xtr,ytr,eval_set=eval_set)
    eval_result = bst.evals_result()
    ypred = bst.predict(xte)
    print (confusion_matrix(yte,ypred))
    print (classification_report(yte,ypred))
    print('The Accuracy is:  ', "{:.2%}".format(accuracy_score(yte, ypred)))
    gen_plot(eval_result)
    return 1
    
    
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
        gsearch4 = GridSearchCV(estimator=xgb_kl(learning_rate=0.1, n_estimators=500, max_depth=3,
                                                        min_child_weight=.001, gamma=0.3, subsample=0.8, colsample_bytree=0.8,
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
        gsearch5 = GridSearchCV(estimator=xgb_kl(learning_rate=0.1, n_estimators=500, max_depth=1,
                                                 min_child_weight=1, gamma=1.03, subsample=0.6, colsample_bytree=0.8,
                                                 objective='binary:logistic', scale_pos_weight=1,
                                                 seed=27, refit=False),
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
                                                 min_child_weight=1, gamma=0.1, subsample=0.6, colsample_bytree=0.6,
                                                 objective='binary:logistic', scale_pos_weight=1,
                                                 seed=27, refit=True),param_grid=param_test7, scoring='roc_auc', cv=5)
        gsearch7.fit(xtr, pdcol2np(ytr))
        gsearch7.best_params_, gsearch7.best_score_
'''