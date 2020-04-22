import argparse
####setup parameters
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type=str, default=
                        './dataset/club_churn_source.csv')
    parser.add_argument("--shuffle", type=int, default=5)
    parser.add_argument("--plt_learn_cv", type=bool, default=True)
    parser.add_argument("--trte_split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--Kfolds", type=int, default=1)
    parser.add_argument("--xgb_param", type=dict, default={
                        'max_depth': 2,
                        'learning_rate': 0.1,
                        'n_estimators': 200,
                        'reg_lambda':0.25,
                        'reg_alpha':0.25})
    parser.add_argument("--xgb_gs_param", type=dict,default={
                        'max_depth': [15,20,25,30,35,40],
                        'learning_rate': [0.05,0.1,0.25],
                        'n_estimators': [16,24],
                        'reg_lambda': [0.125],
                        'reg_alpha': [0.125]})
    parser.add_argument('--estim_list',type=list,default=['xgb'])
    parser.add_argument('--xgb', type=dict, default={'func':'XGBoost',
                                                     'dstruct':'DMatrix',
                                                     'setup':'fill me later',
                                                     'do_train': 'xgb.train',
                                                     'do_cv':   'xgb.cv',
                                                     'get_dataset': 'xgb.DMatrix',
                                                     'rounds': 50,
                                                     'param': {'booster_': 'dart',
                                                               'max_depth':20,
                                                               'learning_rate':0.01,
                                                               'objective':'binary:logistic',
                                                               'gamma': 10,
                                                               'n_estimators': 6000,
                                                               'eval_metric': ['logloss'],
                                                               'min_child_weight_':1,
                                                               'subsample_':0.8,
                                                               'colsample_bytree_':0.8,
                                                               'scale_pos_weight_':1,
                                                               'ntread_':4,
                                                               'reg_alpha_': 5,
                                                               'reg_lambd_': 5},
                                                     'params': {'booster': 'dart',
                                                               'max_depth': 2,
                                                               'learning_rate': 0.1,
                                                               'objective': 'binary:logistic',
                                                               'sample_type': 'uniform',
                                                               'normalize_type': 'tree',
                                                               'rate_drop': 2,
                                                               'skip_drop': 2,
                                                               'num_boost_round': 200,
                                                               'metrics': ['errr', 'rmse']}})



    return parser.parse_args()