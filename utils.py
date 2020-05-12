from args_pg import parse_args
args = parse_args()
import pandas as pd
from sklearn.model_selection import StratifiedKFold,GridSearchCV
import numpy as np
from sklearn.utils  import shuffle
import random


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

def do_grid_srch (mod,x,y,param):
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
    #kfold = KFold(n_splits=4, shuffle=True, random_state=7)
    grid_search = GridSearchCV(mod, param, scoring="recall", n_jobs=-1, cv=kfold, verbose=1)
    grid_result = grid_search.fit(x,y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return grid_result.best_params_

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

from args_pg import *
args = parse_args()
from matplotlib import pyplot
from xgboost import XGBClassifier as xgb_kl
from sklearn.metrics import recall_score
from datetime import datetime
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import learning_curve as lc

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')
import os

#--------------------------------------------------------------------------------------------------------------
def audio_alert (n):
    for i in range(n):
        os.system('afplay /System/Library/Sounds/Sosumi.aiff')
    return

def flatten_list (l):
    return [item for sublist in l for item in sublist]

def gen_cv_plot(dict,dic,rec):
    filename_marker = ''
    audio_alert(1)
    if rec:
        filename_marker = '********'
        audio_alert(2)
    msg = "\n".join("{}\t{}".format(k, v) for k, v in dic.items())
    print(msg)
    tr_crv_dict, te_crv_dict = dict.items()
    tr_crv = tr_crv_dict[1]
    tr_crv_dt = 1 - np.array(tr_crv['recall'])
    te_crv = te_crv_dict[1]
    te_crv_dt = 1 - np.array(te_crv['recall'])
    x = np.arange(1, len(te_crv_dt) + 1)
    fig, ax = pyplot.subplots(figsize=(9, 9))
    bottom, top = pyplot.ylim()
    pyplot.ylim (top = 1.0)
    pyplot.ylim (bottom = 0.0)
    pyplot.grid(b=True, which='both', axis='both')
    ax.plot(x,tr_crv_dt, label='Train')
    ax.plot(x,te_crv_dt, label='Test')
    ax.legend()
    pyplot.title('Train vs. Test')
    ax.text(0.05, 0.95, msg, transform=ax.transAxes, fontsize=9,
            verticalalignment='top')
    pfile = args.plt_dir+'/'+ datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p") + filename_marker
    pyplot.savefig(pfile,dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
    return


def gen_lc_plot(x,y,mod):
    train_sizes,train_scores, test_scores = lc(mod,x,y,train_sizes=np.linspace(0.1, 1.0 , num=10),cv=4)

    pyplot.figure()
    pyplot.title('Learning Curve')
    #if ylim is not None:
        #plt.ylim(*ylim)
    pyplot.xlabel("Training examples")
    pyplot.ylabel("Score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    pyplot.grid()

    pyplot.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    pyplot.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    pyplot.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    pyplot.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    pyplot.legend(loc="best")
    pyplot.show()
    return
