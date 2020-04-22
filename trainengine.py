from args_pg import parse_args
args = parse_args()
import pandas as pd
from sklearn.utils  import shuffle
from transform import transform_,procss
import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV,validation_curve,learning_curve
from sklearn.metrics import confusion_matrix, mean_squared_error,accuracy_score,classification_report
from gen_plots import gen_plot
from xgboost import XGBClassifier as xgb
from sklearn.model_selection import StratifiedKFold

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


def train_ (trte_pd):
    print()
    for estimator in args.estim_list:
        labels = trte_pd.columns[-1]
        y = pdcol2np(procss(trte_pd [[labels]]))
        x = pdcol2np(procss(trte_pd.drop(labels, axis=1)))
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=args.trte_split, random_state=42)
        np.random.shuffle(ytr)
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
        exit()
        print ()
    return 1

def extract_ds():
    pd_master = get_master_pd(args.src_file)
    trte_lst = exp_shuf_fold (pd_master)
    return trte_lst

def main():
    trte_pd= extract_ds()
    print ('***********     Dataset loading was successful')
    for frame in trte_pd:
        train_ (frame)
if __name__ == "__main__":
    main()


'''
        
        bst = do_train(params, dtrain)
        preds = bst.predict(dtest, output_margin=True)
        result = do_cv(params,dtrain,nfold=5,num_boost_round=40,metrics=['error','rmse','rmsle','mae','logloss','auc','aucpr','map'])
        best_nrounds = result.shape[0] - 1
        bst = do_train(params, dtrain, best_nrounds)
        preds = bst.predict(dtest,output_margin=True,objective='binary:logistic')
        ytr = pdcol2np(ytr)
        yte = pdcol2np(yte)
        print (confusion_matrix(yte,preds))
        preds = [round(value) for value in preds]
        accuracy = accuracy_score(yte, preds)



        print()
    return









def plt_learn_cv (X,y,estimator):
    set_size = X.shape[0]
    tr_size = (4*set_size)/5
    train_sizes = np.arange(0,tr_size,round(tr_size/100))
    tr_sizes, train_sc, validation_sc = learning_curve(
        estimator=estimator,
        X=X,y=y,
        train_sizes = train_sizes,
        scoring='neg_mean_squared_error')
    return





        print(('***********     Begin Evaluation'))
        print('   mse      is :  ', "{:.2%}".format(mean_squared_error(Yte, Y_pred)))
        print('   accuracy is :  ', "{:.2%}".format(accuracy_score(Yte, Y_pred)))
        print('************  report  ************  ')
        print(classification_report(Yte, Y_pred))
        plot_results(xgb_trmodel)

        #clf = GridSearchCV(estimator = xgb_m,param_grid=args.xgb_gs_param,scoring='accuracy', verbose=0)
        #clf.fit (Xte,Yte)
        #print('The Best Achieved Accuracy Is:  ',clf.best_score_)
        #print(clf.best_params_)

'''