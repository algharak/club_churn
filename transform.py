import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder,RobustScaler,KBinsDiscretizer
import numpy as np
std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
lbl_encoder = LabelEncoder()
rob_scaler = RobustScaler()
Kbins_disc= KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

#imputes
def most_f_imp(col):
    mode = col.mode().values[0][0]
    newcol = col.fillna(mode)
    return newcol

def avg_imp(col):
    colmean = col.mean().values[0]
    newcol = col.fillna(colmean)
    return newcol

def date2num (item):
    yr = int(item[:4])
    mo = int(item[4:6])
    day = int(item[6:])
    return yr * 365 + mo * 30 + day

def numerize(col):
    nrows = col.shape[0]
    colnp=col.values.flatten().astype(str)
    vfunc = np.vectorize(date2num)
    nu_col = pd.DataFrame(vfunc(colnp).reshape(nrows,1), index=col.index, columns=col.columns)
    return nu_col

scale_ = {'std_sc': std_scaler.fit_transform,
               'mm_sc' : mm_scaler.fit_transform,
               'rob_sc': rob_scaler.fit_transform,
               'lbl_enc_sc':lbl_encoder.fit_transform,
               'k_bin_disc_sc':Kbins_disc.fit_transform}

impute_ = {'avg_imp':avg_imp,
           'most_f_imp':most_f_imp,
           'numerize':numerize}

#--------------------------------------------------------------------------------------------------
col_assign = {'MEMBERSHIP_STATUS':[[None],['lbl_enc_sc']]}
#--------------------------------------------------------------------------------------------------
col_assign.update({'MEMBERSHIP_TERM_YEARS':[[None],['std_sc','mm_sc','rob_sc']]})
#col_assign.update({'MEMBERSHIP_TERM_YEARS':[[None],['std_sc','mm_sc','rob_sc','k_bin_disc_sc']]})
#col_assign.update({'MEMBERSHIP_TERM_YEARS':[[None],['std_sc']]})
#--------------------------------------------------------------------------------------------------
col_assign.update({'ANNUAL_FEES':[[None],['std_sc']]})
#--------------------------------------------------------------------------------------------------
col_assign.update({'MEMBER_MARITAL_STATUS':[['most_f_imp'],['lbl_enc_sc']]})
#--------------------------------------------------------------------------------------------------
col_assign.update({'MEMBER_GENDER':[['most_f_imp'],['lbl_enc_sc']]})
#--------------------------------------------------------------------------------------------------
#col_assign.update({'MEMBER_ANNUAL_INCOME':[['avg_imp'],['std_sc']]})
col_assign.update({'MEMBER_ANNUAL_INCOME':[['avg_imp'],['std_sc','mm_sc','rob_sc']]})
#col_assign.update({'MEMBER_ANNUAL_INCOME':[['avg_imp'],['std_sc','mm_sc','rob_sc','k_bin_disc_sc']]})
#--------------------------------------------------------------------------------------------------
col_assign.update({'MEMBER_OCCUPATION_CD':[[None],['lbl_enc_sc']]})
#--------------------------------------------------------------------------------------------------
#col_assign.update({'MEMBERSHIP_PACKAGE':[[None],[None]]})
col_assign.update({'MEMBERSHIP_PACKAGE':[[None],['lbl_enc_sc']]})
#--------------------------------------------------------------------------------------------------
#col_assign.update({'MEMBER_AGE_AT_ISSUE':[[None],['std_sc']]})
col_assign.update({'MEMBER_AGE_AT_ISSUE':[[None],['std_sc','mm_sc','rob_sc']]})
#col_assign.update({'MEMBER_AGE_AT_ISSUE':[[None],['std_sc','mm_sc','rob_sc','k_bin_disc_sc']]})
#--------------------------------------------------------------------------------------------------
col_assign.update({'ADDITIONAL_MEMBERS':[[None],['lbl_enc_sc']]})
#--------------------------------------------------------------------------------------------------
col_assign.update({'PAYMENT_MODE':[[None],['lbl_enc_sc']]})
#--------------------------------------------------------------------------------------------------
#col_assign.update({'START_DATE':[['numerize'],['std_sc']]})
col_assign.update({'START_DATE':[['numerize'],['std_sc','mm_sc','rob_sc']]})
#col_assign.update({'START_DATE':[['numerize'],['std_sc','mm_sc','rob_sc','k_bin_disc_sc']]})
#col_assign.update({'START_DATE':[['numerize'],['std_sc','mm_sc']]})
#--------------------------------------------------------------------------------------------------
col_assign.update({'INDEX':[[None],[None]]})
col_assign.update({'MEMBERSHIP_NUMBER':[[None],[None]]})
col_assign.update({'AGENT_CODE':[[None],[None]]})
col_assign.update({'END_DATE':[[None],[None]]})
#--------------------------------------------------------------------------------------------------


impute_ = {'avg_imp':avg_imp,
           'most_f_imp':most_f_imp,
           'numerize':numerize}

def procss_impute (clm,cmd):
    nu_clm=impute_[cmd[0]](clm)
    return nu_clm

def procss_scale (clm,cmd,frm):
    colname = clm.columns[0]
    colnuname = colname + '_' + cmd
    colvals = clm[colname].values
    if cmd != 'lbl_enc_sc':
        colvals = colvals.reshape(len(colvals), 1)
    col_np =scale_[cmd](colvals)
    clm_o =pd.DataFrame(col_np,columns=[colnuname])
    frm_o = pd.concat([frm,clm_o],axis=1)
    return frm_o

def procss (frm):
    out_frame = pd.DataFrame()
    active_cols = list(col_assign.keys())
    colnames = frm.columns.values.tolist()
    for inx in colnames:
        if inx in active_cols:
            imput_cmd = col_assign[inx][0]
            scale_cmds = col_assign[inx][1]
            col = frm [[inx]]
            if imput_cmd != [None]:
                col     =   procss_impute(col,imput_cmd)
            if scale_cmds !=[None]:
                for scale_cmd in scale_cmds:
                    out_frame = procss_scale(col,scale_cmd,out_frame)
    return out_frame

def transform_(trte_list):
    tr_te_plist = []
    for pair in trte_list:
        tr_te_pair = []
        for frm in pair:
            tr_te_pair.extend([procss(frm)])
        tr_te_plist.append(tr_te_pair)
    return tr_te_plist



