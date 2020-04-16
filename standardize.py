#import argparse
#import json
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
import numpy as np
std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
lbl_encoder = LabelEncoder()

colnames =          ['MEMBERSHIP_NUMBER',
                    'MEMBERSHIP_TERM_YEARS',
                    'ANNUAL_FEES',
                    'MEMBER_MARITAL_STATUS',
                    'MEMBER_GENDER',
                    'MEMBER_ANNUAL_INCOME',
                    'MEMBER_OCCUPATION_CD',
                    'MEMBERSHIP_PACKAGE',
                    'MEMBER_AGE_AT_ISSUE',
                    'ADDITIONAL_MEMBERS',
                    'PAYMENT_MODE',
                    'AGENT_CODE',
                    'START_DATE',
                    'END_DATE',
                    'MEMBERSHIP_STATUS']

newcolnames=['MEMBERSHIP_STATUS',
              'MEMBERSHIP_TERM_YEARS_MINMAX_SC',
             'MEMBERSHIP_TERM_YEARS_STD_SC',
             'ANNUAL_FEES_MINMAX_SC',
             'ANNUAL_FEES_STD_SC',
             'MEMBER_MARITAL_STATUS_MOST_CAT',
             'MEMBER_GENDER_MOST_CAT',
             'MEMBER_ANNUAL_INCOME_AVG_MINMAX_SC',
              'MEMBER_ANNUAL_INCOME_AVG_STD_SC',
             'MEMBER_OCCUPATION_CD_CAT',
             'MEMBERSHIP_PACKAGE_CAT',
             'MEMBER_AGE_AT_ISSUE_MINMAX_SC',
             'ADDITIONAL_MEMBERS_CAT',
             'PAYMENT_MODE_CAT',
             'START_DATE_MINMAX_SC',
             'END_DATE_MINMAX_SC']
#### setup parameters
ds_len = 'Long'
ds_type =   'Train'

is_short = True if ds_len == 'Short' else False
is_long = not is_short
is_tr   = True if ds_type == 'Train' else False
is_te   = not is_tr

InFilenames       = ['./dataset/club_churn_test_short.csv','./dataset/club_churn_test.csv','./dataset/club_churn_train_short.csv','./dataset/club_churn_train.csv',]
OutFilenames       = ['./dataset/club_churn_test_short_std.csv','./dataset/club_churn_test_std.csv','./dataset/club_churn_train_short_std.csv','./dataset/club_churn_train_std.csv']
ptr = 2*is_tr + is_long
dataset_in      = InFilenames [ptr]
dataset_out      = OutFilenames [ptr]

###### utilities
def clear_old_outputs (file):
    if os.path.isfile(file):
        os.remove(file)
        print("Old file cleared")
    else:
        print("Directory Clear")
    return

def fill_blanks(df,item):
    return df.fillna (item)


def date2int(a):
    astr = str(round(a))
    yr = int(astr[:4])
    mo = int(astr[4:6])
    day = int(astr[6:])
    return yr*365+mo*30+day

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def append_r(col,pd_o):
    pd_out = pd.concat([pd_o,col],axis=1)
    return pd_out

def append_l(col,pd_o):
    pd_out = pd.concat([col,pd_o],axis=1)
    return pd_out

#imputes
def most_often_imp(col):
    mode = col.mode().as_matrix()[0,0]
    col.fillna(value=mode,inplace=True)
    return col

def avg_imp(col):
    col.fillna(col.mean(), inplace=True)
    return col



#scalers
def minmax_scaler(col):
    colname = col.columns[0]
    colvals= col[colname].values
    colvals = colvals.reshape(len(colvals), 1)
    mm_scaler.fit(colvals)
    col.loc[:,colname] = mm_scaler.transform(colvals)
    return col

#functions
def std_scale(col):
    colname = col.columns
    std_scaler.fit(col[colname])
    col[colname] = std_scaler.transform(col[colname])
    return col

def categorize(col):
    colname = col.columns[0]
    col_list= col[colname].tolist()
    lbl_encoder.fit(col_list)
    print(lbl_encoder.classes_)
    pd_o = pd.DataFrame({colname:lbl_encoder.transform(col_list)})
    return pd_o

def mem_status_cat(pd_i,pd_o):
    src_col = 'MEMBERSHIP_STATUS'
    tgt_col =   'MEMBERSHIP_STATUS'
    pd_o [tgt_col] = pd_i [src_col]
    pd_o[tgt_col] = categorize(pd_o [[tgt_col]])
    return pd_o
#-----------------------------------------------------------
def mem_term_yrs_mm_sc(pd_i,pd_o):


    src_col = 'MEMBERSHIP_TERM_YEARS'
    tgt_col = 'MEMBERSHIP_TERM_YEARS_MINMAX_SC'
    pd_o[tgt_col]=minmax_scaler(pd_i [[src_col]])
    return pd_o

def mem_term_yrs_std_sc(pd_i,pd_o):

    src_col = 'MEMBERSHIP_TERM_YEARS'
    tgt_col = 'MEMBERSHIP_TERM_YEARS_STD_SC'
    pd_o[tgt_col] = std_scale(pd_i[[src_col]])
    return pd_o

def ann_fee_mm_sc(pd_i,pd_o):
    src_col = 'ANNUAL_FEES'
    tgt_col = 'ANNUAL_FEES_MINMAX_SC'
    pd_o[tgt_col] = minmax_scaler(pd_i[[src_col]])
    return pd_o

def ann_fee_std_sc(pd_i,pd_o):
    src_col = 'ANNUAL_FEES'
    tgt_col = 'ANNUAL_FEES_STD_SC'
    pd_o[tgt_col] = std_scale(pd_i[[src_col]])
    return pd_o

def mem_marstat_most_cat(pd_i,pd_o):
    src_col = 'MEMBER_MARITAL_STATUS'
    tgt_col = 'MEMBER_MARITAL_STATUS_MOST_CAT'
    imputed_pd = most_often_imp(pd_i[[src_col]])
    pd_o[tgt_col]= categorize(imputed_pd)
    return pd_o

def mem_gen_most_cat(pd_i,pd_o):
    src_col = 'MEMBER_GENDER'
    tgt_col = 'MEMBER_GENDER_MOST_CAT'
    imputed_pd = most_often_imp(pd_i[[src_col]])
    pd_o[tgt_col] = categorize(imputed_pd)
    return pd_o

def mem_inc_avg_mm_sc(pd_i,pd_o):
    src_col = 'MEMBER_ANNUAL_INCOME'
    tgt_col = 'MEMBER_ANNUAL_INCOME_AVG_MINMAX_SC'
    imputed_pd = avg_imp(pd_i[[src_col]])
    pd_o[tgt_col] = minmax_scaler(imputed_pd)
    return pd_o

def mem_inc_avg_std_sc(pd_i,pd_o):
    src_col = 'MEMBER_ANNUAL_INCOME'
    tgt_col = 'MEMBER_ANNUAL_INCOME_AVG_STD_SC'
    imputed_pd = avg_imp(pd_i[[src_col]])
    pd_o[tgt_col] = std_scale(imputed_pd[[src_col]])
    return pd_o

def mem_occ_cat(pd_i,pd_o):
    src_col = 'MEMBER_OCCUPATION_CD'
    tgt_col = 'MEMBER_OCCUPATION_CD_CAT'
    imputed_pd = most_often_imp(pd_i[[src_col]])
    pd_o[tgt_col] = categorize(imputed_pd[[src_col]])
    return pd_o

def mem_pkg_cat(pd_i,pd_o):
    src_col = 'MEMBERSHIP_PACKAGE'
    tgt_col = 'MEMBERSHIP_PACKAGE_CAT'
    pd_o[tgt_col] = categorize(pd_i[[src_col]])
    return pd_o

def mem_age_mm_sc(pd_i,pd_o):
    src_col = 'MEMBER_AGE_AT_ISSUE'
    tgt_col = 'MEMBER_AGE_AT_ISSUE_MINMAX_SC'
    pd_o[tgt_col] = minmax_scaler(pd_i[[src_col]])
    return pd_o

def mem_addmem_mm_sc(pd_i,pd_o):
    src_col = 'ADDITIONAL_MEMBERS'
    tgt_col = 'ADDITIONAL_MEMBERS_CAT'
    pd_o[tgt_col] = categorize(pd_i[[src_col]])
    return pd_o

def mem_paymode_cat(pd_i,pd_o):
    src_col = 'PAYMENT_MODE'
    tgt_col = 'PAYMENT_MODE_CAT'
    pd_o[tgt_col] = categorize(pd_i[[src_col]])
    return pd_o

def mem_stdate_mm_sc(pd_i,pd_o):
    src_col = 'START_DATE'
    tgt_col = 'START_DATE_MINMAX_SC'
    st_datesn = pd_i[src_col].apply (date2int)
    st_datesnp = st_datesn.values
    st_datesnp = st_datesnp.reshape(len(st_datesnp), 1)
    scaler = mm_scaler.fit(st_datesnp)
    pd_o[tgt_col] = mm_scaler.transform(st_datesnp)
    return pd_o

def mem_endt_mm_sc(pd_i,pd_o):
    src_col = 'END_DATE'
    tgt_col = 'END_DATE_MINMAX_SC'
    pd_o.drop(columns=[tgt_col],inplace=True)
    return pd_o

def mem_dur_mm_sc(pd_i,pd_o):
    src_col = 'MEMBER_OCCUPATION_CD'
    tgt_col = 'MEMBER_OCCUPATION_CD_CAT'
    return pd_o


def pre_processor(pin,cn,pout):
    if cn==0:
        pout = mem_status_cat (pin,pout)
    if cn == 1:
        pout = mem_term_yrs_mm_sc(pin,pout) #
    elif cn== 2:
        pout = mem_term_yrs_std_sc(pin,pout) #
    elif cn== 3:
        pout = ann_fee_mm_sc(pin,pout) #
    elif cn== 4:
        pout = ann_fee_std_sc(pin,pout) #
    elif cn== 5:
        pout = mem_marstat_most_cat(pin,pout) #
    elif cn== 6:
        pout = mem_gen_most_cat(pin,pout) #
    elif cn== 7:
        pout = mem_inc_avg_mm_sc(pin,pout) #
    elif cn== 8:
        pout = mem_inc_avg_std_sc(pin,pout) #
    elif cn== 9:
        pout = mem_occ_cat(pin,pout) #
    elif cn == 10:
        pout = mem_pkg_cat(pin, pout) #
    if cn == 11:
        pout = mem_age_mm_sc(pin,pout) #
    elif cn == 12:
        pout = mem_addmem_mm_sc(pin,pout) #
    elif cn == 13:
        pout = mem_paymode_cat(pin,pout)
    elif cn== 14:
        pout = mem_stdate_mm_sc(pin,pout)
    elif cn== 15:
        pout=   mem_endt_mm_sc (pin,pout)
    return pout

###### retrieve input dataset
clear_old_outputs(dataset_out)
print(dataset_in)
pd_dsetin =pd.read_csv(dataset_in)
print (pd_dsetin.dtypes)
nrows=pd_dsetin.shape[0]
ncols=pd_dsetin.shape[1]
n_newcols:len(newcolnames)
pd_dsetout =pd.DataFrame(index=np.arange(nrows),columns=newcolnames)
print (ncols, '   ', nrows)

##### Main Loop
for coln,col in enumerate(newcolnames):
   print (coln,col)
   p_out = pre_processor(pd_dsetin,coln,pd_dsetout)
p_out.to_csv(dataset_out,index=False)
exit()











