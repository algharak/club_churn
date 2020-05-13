from args_pg import args
from sklearn.metrics import confusion_matrix, balanced_accuracy_score,accuracy_score,classification_report,precision_score,recall_score
from sklearn.model_selection import StratifiedKFold
import os
from hyper_opt import *
from xgboost import plot_importance

from sklearn.model_selection import KFold, train_test_split, GridSearchCV,validation_curve,learning_curve
from args_pg import parse_args
args = parse_args()
from transform import *
from utils import *

class dset():
    def __init__(self,frm,clip=False,clip_size=2000,shuffle=True):
        self.labels = frm.columns[-1]
        self.predictors = frm.columns[0:-1]
        self.frm = pd.concat([frm,frm,frm,frm,frm],ignore_index=True)
        #self.frm = frm
        nufrm = self.frm.copy()
        if clip:
            nufrm = self.frm.head(clip_size)
        if shuffle:
            nufrm = myshuffle(nufrm)
        self.split_xy(nufrm)
    def xtr (self,stdiz=True,np = False):
        x = self.xtra
        if stdiz:
            x = procss(x)
        if np:
            x = pdcol2np(x)
        self.xtr = x
        return x
    def xte (self,stdiz=True,np = False):
        x=self.xtes
        if stdiz:
            x = procss(x)
        if np:
            x = pdcol2np(x)
        self.xte = x
        return x
    def yte (self,stdiz=True,np = True,yshuff = False):
        y = self.ytes
        y = y.reset_index(drop=True)
        if yshuff:
            y = myshuffle(y)
        if stdiz:
            y = procss(y).astype(bool)
            y=~y
            print ('dist of yte is:', y.sum()/y.shape[0])
        if np:
            y = pdcol2np(y)
        self.yte = y
        return y
    def ytr (self,stdiz=True,np = True):
        y = self.ytra
        if stdiz:
            y = procss(y).astype(bool)
            y = ~y
            print ('dist of ytr is:', y.sum()/y.shape[0])
        if np:
            y = pdcol2np(y)
        self.ytr = y
        return y
    def split_xy(self,f):
        y = f[[self.labels]]
        x = f.drop(self.labels, axis=1)
        self.xtra, self.xtes, self.ytra, self.ytes = train_test_split(x, y, test_size=args.trte_split, random_state=42)

def train_ (scn):
    print ('***********     Start The Training Process')
    xtr=scn.xtr() ;xte=scn.xte() ;ytr=scn.ytr() ;yte=scn.yte()
    eval_set = [(xtr,ytr),(xte,yte)]
    best_recall_store = 0
    best_par = 0
    for round in range(args.exp_rounds):
        best_par = args.base_param
        mod = xgb_kl(**best_par)
        if args.param_rng:
            best_par,best_val=tune_params (scn)
            mod = xgb_kl(**best_par)
        best_mod = mod.fit(xtr, ytr, eval_set=eval_set,eval_metric=metric_recall,verbose=True)
        plot_importance(best_mod)
        eval_result = best_mod.evals_result()
        ypred = best_mod.predict(xte)
        #print(confusion_matrix(yte, ypred))
        #print(classification_report(yte, ypred))
        print('The Accuracy is:  ', "{:.2%}".format(accuracy_score(yte, ypred)))
        print('The Balanced Accuracy is:  ', "{:.2%}".format(balanced_accuracy_score(yte, ypred)))
        print('The Precision Score is:  ', "{:.2%}".format(precision_score(yte, ypred)))
        recall_scr = recall_score(yte, ypred)
        print('The Recall score is:  ', "{:.2%}".format(recall_scr))
        record_set = False
        if recall_scr > best_recall_store:
            print ('the best recall score so far is:  ', recall_scr)
            print('the corresponding params are:  ', best_par)
            best_recall_store = recall_scr
            record_set = True
        gen_cv_plot(eval_result,best_par,record_set)
        #gen_lc_plot(xtr,ytr,best_mod)
    print ('***********Experiment Completed************')
    return

def metric_recall(y_pred, y_true):
    labels = y_true.get_label() # obtain true labels
    preds = y_pred > 0.5 # obtain predicted values
    preds=preds.astype(np.int)
    recall_not = 1-recall_score(labels,preds)
    return 'recall', recall_not

def main():
    trte_pd= extract_ds()
    run_plan = dset (trte_pd)
    train_ (run_plan)
if __name__ == "__main__":
    main()
