from args_pg import *
args = parse_args()
from matplotlib import pyplot
from experiment import *
from xgboost import XGBClassifier as xgb_kl
from sklearn.metrics import recall_score
from datetime import datetime
from matplotlib.font_manager import FontProperties

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')

#--------------------------------------------------------------------------------------------------------------
def flatten_list (l):
    return [item for sublist in l for item in sublist]

def gen_plot(dict,dic):
    msg = "\n".join("{}\t{}".format(k, v) for k, v in dic.items())
    tr_crv_dict,te_crv_dict = dict.items()
    tr_crv  = tr_crv_dict[1]
    tr_crv_dt  = 1-np.array(tr_crv['recall'])
    te_crv = te_crv_dict[1]
    te_crv_dt = 1- np.array(te_crv['recall'])
    x = np.arange(1,len(te_crv_dt)+1)
    fig, ax = pyplot.subplots(figsize=(9, 9))
    bottom, top = pyplot.ylim()
    pyplot.ylim (top = 1.0)
    pyplot.ylim (bottom = 0.0)
    pyplot.grid(b=True, which='both', axis='both')
    ax.plot(x,tr_crv_dt, label='Train')
    ax.plot(x,te_crv_dt, label='Test')
    ax.legend()
    pyplot.title('Train vs. Test Recall')
    ax.text(0.05, 0.95, msg, transform=ax.transAxes, fontsize=9,
            verticalalignment='top')
    pfile = args.plt_dir+'/'+ datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
    pyplot.savefig(pfile,dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
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
