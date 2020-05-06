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
    #xgb params
    baseparam=dict(objective='binary:logistic')
    #baseparam.update(dict(n_estimators=500))
    baseparam.update(dict(learning_rate=0.1))
    baseparam.update(dict(scale_pos_weight=1))
    #baseparam.update(dict(max_depth=4))
    #baseparam.update(dict(min_child_weight=5))
    #baseparam.update(dict(gamma=0.3))
    #baseparam.update(dict(subsample=0.6))
    #baseparam.update(dict(colsample_bytree=0.7))
    parser.add_argument("--base_param", type=dict, default=baseparam )
    #Ax params
    parser.add_argument("--max_eval", type=int, default=50)
    return parser.parse_args()

args=argparse
'''
Notes
scale_pos_weights: be very carfull. large numbers screw things up.  i used the formaula r
recommended but it does not work.  will experiment with value 1
'''

