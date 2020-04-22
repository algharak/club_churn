from args_pg import parse_args
args = parse_args()
import pandas as pd
import numpy as np

def pdcol2np (col):
    nrows = col.shape[0]
    ncols = col.shape[1]
    colnp = col.values.reshape((nrows,ncols))
    if ncols==1:
        colnp = col.values.reshape((nrows))
    return colnp

def build_dict(lst):
    mydict=dict()
    for item in lst:
        mydict[item]={'trcol':'train-'+item+'-mean',
                      'tecol':'test-'+item+'-mean',
                      'ylab': item,
                      'title': item}
    return mydict
