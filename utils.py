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
'''