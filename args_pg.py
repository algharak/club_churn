import argparse
####setup parameters
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type=str, default=
                        './dataset/club_churn_source.csv')
    parser.add_argument("--shuffle", type=int, default=2)
    parser.add_argument("--plt_learn_cv", type=bool, default=True)
    parser.add_argument("--trte_split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--Kfolds", type=int, default=1)
    parser.add_argument("--xgb_gs_param", type=dict,default={
                        'max_depth': [15,20,25,30,35,40],
                        'learning_rate': [0.05,0.1,0.25],
                        'n_estimators': [16,24],
                        'reg_lambda': [0.125],
                        'reg_alpha': [0.125]})
    parser.add_argument('--xgb_sk', type=dict, default=
                        {   "max_depth":4,
                            "learning_rate":0.002,
                            "objective":'binary:logistic',
                            #"n_estimators":100,
                            #'num_boost_round':100,
                            "booster":'dart',
                            #"gamma":0.1,
                            #"min_child_weight":1,
                            #"subsample":1,
                            #"colsample_bytree":0.8,
                            #"reg_alpha":0.9,
                            #"reg_lambda":0.9,
                            #"scale_pos_weight":0.41
                                                       })
    parser.add_argument('--maxdepth_grid',type = list, default = [2,4,6,8,10,20,40,80])
    parser.add_argument('--estimator_grid',type = list, default = [10,15,20,25,40,60,80,100,120])
    return parser.parse_args()

args=argparse




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