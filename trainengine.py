from args_pg import parse_args
args = parse_args()
import pandas as pd
from sklearn.utils  import shuffle
from transform import transform_
import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV,validation_curve
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, mean_squared_error,accuracy_score,classification_report

def load_datasets(trte_lst):
    X_tr = [];Y_tr = [];X_te = [];Y_te = []
    flat_list = [item for sublist in trte_lst for item in sublist]
    trlst, telst = flat_list[::2],flat_list[1::2]
    for i in range(len(trlst)):
        Y_tr.append(trlst[i].iloc[:,0].values.astype(int))
        X_tr.append(trlst[i].iloc[:,1:].values.astype(float))
        Y_te.append(telst[i].iloc[:, 0].values.astype(int))
        X_te.append(telst[i].iloc[:, 1:].values.astype(float))
    return zip(X_tr,Y_tr,X_te,Y_te)

def get_master_pd(path):
    return pd.read_csv(path)

def exp_shuf_fold(fr):
    newfrm = fr.copy()
    for iter in range(args.shuffle):
        newfrm = shuffle(newfrm)
    kf = KFold(n_splits=args.Kfolds)
    te_tr_list = []
    for train, test in kf.split(newfrm):
        tr_epochs = args.epochs*[newfrm.iloc[train]]
        tr_epochs = pd.concat(tr_epochs)
        tr_epochs = shuffle(tr_epochs)
        te_tr_list.append([tr_epochs]+[newfrm.iloc[test]])
    return te_tr_list

def plot_results (mod):
    results = mod.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
    ax.legend()
    pyplot.ylabel('RMSE')
    pyplot.title('Train/Test RMSE')
    pyplot.show()
    # plot classification error
    '''
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    pyplot.ylabel('Classification Error')
    pyplot.title('Classification Error')
    pyplot.show()
    '''
    return

def train_evaluate (tr_ev_dt):
    for Xtr,Ytr,Xte,Yte in tr_ev_dt:
        eval_set = [(Xtr,Ytr),(Xte,Yte)]
        print(('***********     Begin Training'))
        xgb_m = xgb.XGBClassifier()
        xgb_model = xgb.XGBClassifier(**args.xgb_param)
        xgb_trmodel =xgb_model.fit(Xtr, Ytr,eval_metric=["error", "rmse"], eval_set=eval_set, verbose=True)
        Y_pred = xgb_trmodel.predict(Xte)
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

def extract_ds():
    pd_master = get_master_pd(args.src_file)
    trte_lst = exp_shuf_fold (pd_master)
    return trte_lst

def main():
    trte_lst= extract_ds()
    trte_proc_lst = transform_ (trte_lst)
    print ('***********    Dataset Preprocessing was successful')
    tr_eval_data = load_datasets(trte_proc_lst)
    print ('***********     Dataset loading was successful')
    train_evaluate (tr_eval_data)

if __name__ == "__main__":
    main()
