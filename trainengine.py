from args_pg import args
from sklearn.metrics import confusion_matrix, balanced_accuracy_score,accuracy_score,classification_report,precision_score,recall_score
from gen_plots import *
from xgboost import XGBClassifier as xgb_kl
from xgboost import plot_importance
import random
from experiment import *
from hyper_opt import *


def get_master_pd(path):
    return pd.read_csv(path)

def extract_cmds (e):
    do_train = eval(eval('args.'+e)['do_train'])
    do_train = eval(eval('args.'+e)['do_train'])
    params =eval('args.'+e)['param']
    rounds =eval('args.'+e)['rounds']
    get_dataset =eval(eval('args.'+e)['get_dataset'])
    do_cv = eval(eval('args.'+e)['do_cv'])
    return do_train,do_cv,rounds,params, get_dataset

def train_ (scn):
    print ('***********     Start The Training Process')
    xtr=scn.xtr() ;xte=scn.xte() ;ytr=scn.ytr() ;yte=scn.yte()
    eval_set = [(xtr,ytr),(xte,yte)]
    best_par,best_val=tune_params (scn)
    print ('***********     Perform Test')
    mod = xgb_kl(**best_par)
    best_mod = mod.fit(xtr, ytr, eval_set=eval_set,eval_metric=metric_recall,verbose=False)
    plot_importance(best_mod)
    pyplot.show()
    eval_result = best_mod.evals_result()
    ypred = best_mod.predict(xte)
    print(confusion_matrix(yte, ypred))
    print(classification_report(yte, ypred))
    print('The Accuracy is:  ', "{:.2%}".format(accuracy_score(yte, ypred)))
    print('The Balanced Accuracy is:  ', "{:.2%}".format(balanced_accuracy_score(yte, ypred)))
    print('The Precision Score is:  ', "{:.2%}".format(precision_score(yte, ypred)))
    print('The Recall score is:  ', "{:.2%}".format(recall_score(yte, ypred)))
    gen_plot(eval_result)
    exit()

def metric_recall(y_pred, y_true):
    labels = y_true.get_label() # obtain true labels
    preds = y_pred > 0.5 # obtain predicted values
    preds=preds.astype(np.int)
    recall = 1-recall_score(labels,preds)
    return 'recall', recall

def do_grid_srch (mod,x,y,param):
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
    kfold = KFold(n_splits=4, shuffle=True, random_state=7)
    grid_search = GridSearchCV(mod, param, scoring="recall", n_jobs=-1, cv=kfold, verbose=1)
    grid_result = grid_search.fit(x,y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return grid_result.best_params_

###########################################################################################################################


def extract_ds():
    pd_master = get_master_pd(args.src_file)
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
    train_ (run_plan)
if __name__ == "__main__":
    main()
