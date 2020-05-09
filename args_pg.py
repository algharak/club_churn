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
    # param baseline
    parser.add_argument("--Ax_max_iter", type=int, default=5)
    parser.add_argument("--Ax_n_trials", type=int, default=5)
    # params
    param_rng=[]
    baseparam = dict(objective='binary:logistic')
    baseparam.update(dict(booster='dart'))
    baseparam.update(dict(n_estimators=50))

    # LR
    baseparam.update(dict(learning_rate=0.1))
    #lr_rng = {"name": "learning_rate","type": "range","bounds": [1e-3,0.9],"value_type": "float",'log_scale':True}
    #param_rng.append(lr_rng)

    # scale_pos_weight
    baseparam.update(dict(scale_pos_weight=1))
    #sc_po_rng = {"name": "scale_pos_weight","type": "range","bounds": [1,1.5],"value_type": "float",'log_scale':False}
    #param_rng.append(sc_po_rng)

    # max_depth
    baseparam.update(dict(max_depth=3))
    #max_de_rng = {"name": "max_depth","type": "range","bounds": [2,6],"value_type": "int",'log_scale':False}
    #param_rng.append(max_de_rng)

    # min_child_weight
    #baseparam.update(dict(min_child_weight=2))
    min_c_we_rng = {"name":'min_child_weight',"type": "range","bounds": [2,6],"value_type": "int",'log_scale':False}
    param_rng.append(min_c_we_rng)

    # gamma
    #baseparam.update(dict(gamma=0.05))
    gamma_rng = {'name':'gamma',"type": "range","bounds": [1e-3,0.9],"value_type": "float",'log_scale':True}
    param_rng.append(gamma_rng)

    # subsample range
    #baseparam.update(dict(subsample=0.8))
    subsam_rng = {'name':'subsample',"type": "choice","values": [0.3,0.5,0.7,0.9],"value_type": "float",'log_scale':False}
    param_rng.append(subsam_rng)

    # colsample
    #baseparam.update(dict(colsample_bytree=0.6))
    colsam_rng = {'name':'colsample_bytree',"type": "choice","values": [0.3,0.5,0.7,0.9],"value_type": "float",'log_scale':False}
    param_rng.append(colsam_rng)

    # reg_alpha
    #baseparam.update(dict(reg_alpha=0.01))
    reg_a_rng = {'name':'reg_alpha',"type": "range","bounds": [1e-3,0.9],"value_type": "float",'log_scale':True}
    param_rng.append(reg_a_rng)

    # Aggregation
    parser.add_argument('--base_param',type=dict,default=baseparam)
    parser.add_argument('--param_rng', type=list, default=param_rng)
    parser.add_argument('--colnames', type=list, default=['objective']+[item['name'] for item in param_rng])
    return parser.parse_args()

args=argparse


