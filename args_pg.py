import argparse
####setup parameters
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type=str, default=
                        './dataset/club_churn_source.csv')
    parser.add_argument("--shuffle", type=int, default=2)
    parser.add_argument("--plt_learn_cv", type=bool, default=True)
    parser.add_argument("--trte_split", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--Kfolds", type=int, default=1)
    parser.add_argument("--xgb_gs_param", type=dict,default={
                        'max_depth': [15,20,25,30,35,40],
                        'learning_rate': [0.05,0.1,0.25],
                        'n_estimators': [16,24],
                        'reg_lambda': [0.125],
                        'reg_alpha': [0.125]})
    parser.add_argument('--estim_list',type=list,default=['xgb'])
    parser.add_argument('--xgb', type=dict, default=
                        {'func':'XGBoost',
                        'dstruct':'DMatrix',
                        'do_train': 'xgb.train',
                        'do_cv':   'xgb.cv',
                        'get_dataset': 'xgb.DMatrix',
                        'rounds': 500,
                        'param': {'booster_': 'dart','max_depth':20,                                         'learning_rate':0.01,
                        'objective':'binary:logistic'},'gamma': 10,
                        'n_estimators': 60000,
                        'eval_metric': ['logloss'],
                            'min_child_weight_':1,
                            'subsample_':0.8,
                            'colsample_bytree_':0.8,
                            'scale_pos_weight_':1,
                            'ntread_':4,
                            'reg_alpha_': 5,'reg_lambd_': 5})


    return parser.parse_args()

parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--learning_rate", type=float, default=0.25)
parser.add_argument("--objective", type=str, default='binary:logistic')
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--objective=", type=str, default='binary:logistic')
parser.add_argument("--booster", type=str, default='dart')
parser.add_argument("--gamma", type=float, default=0.1)

parser.add_argument("--min_child_weight", type=float, default=1)
parser.add_argument("--subsample", type=float, default=1)

parser.add_argument("--trte_split", type=float, default=0.25)
parser.add_argument("--trte_split", type=int, default=0.25)


'''

#sklearn XGBModel
max_depth=5,
learning_rate=0.05,
objective='binary:logistic'
n_estimators
booster 'dart'
gamma=0,
min_child_weight=1,
subsample=0.8,
colsample_bytree=0.8,
reg_alpha
reg_lambda
scale_pos_weight
#fit
eval_set
eval_metric
early_stopping_rounds
#predict


#xgboost.DMatrix
label
feature_names






nthread=4,
scale_pos_weight=0.41
seed=27

#xgb



num_round =
reg_lambda
reg_alpha
'''