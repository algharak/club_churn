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
    #
    parser.add_argument("--en_hp_tune", type=int, default=False)
    parser.add_argument("--Ax_max_iter", type=int, default=5)
    parser.add_argument("--Ax_n_trials", type=int, default=30)
    #
    baseparam = dict(objective='binary:logistic')
    baseparam.update(dict(n_estimators=1000))
    baseparam.update(dict(learning_rate=0.1))
    baseparam.update(dict(scale_pos_weight=1))
    baseparam.update(dict(booster='dart'))
    baseparam.update(dict(max_depth=3))
    baseparam.update(dict(min_child_weight=2))
    baseparam.update(dict(gamma=0.05))
    baseparam.update(dict(subsample=0.8))
    baseparam.update(dict(colsample_bytree=0.6))
    baseparam.update(dict(reg_alpha=0.01))
    parser.add_argument('--base_param',type=dict,default=baseparam)
    #
    lr_rng = {}
    sc_po_rng={}
    max_de_rng ={}
    min_c_we_rng = {}
    gamma_rng = {}
    subsam_rng = {}
    colsam_rng = {}
    reg_a_rng = {}
    param_range_list = [lr_rng,sc_po_rng,max_de_rng,min_c_we_rng,gamma_rng,subsam_rng,colsam_rng,reg_a_rng]
    parser.add_argument('--all_param_rng', type=list, default=param_range_list)



    return parser.parse_args()

args=argparse
'''
Notes
scale_pos_weights: be very carfull. large numbers screw things up.  i used the formaula r
recommended but it does not work.  will experiment with value 1
'''

