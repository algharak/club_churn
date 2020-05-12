from sklearn.model_selection import KFold, train_test_split, GridSearchCV,validation_curve,learning_curve
from args_pg import parse_args
args = parse_args()
from transform import *
from utils import *

class dset():
    def __init__(self,frm,clip=False,clip_size=2000,shuffle=True):
        self.labels = frm.columns[-1]
        self.predictors = frm.columns[0:-1]
        self.frm = frm
        nufrm = self.frm.copy()
        if clip:
            nufrm = self.frm.head(clip_size)
        if shuffle:
            nufrm = myshuffle(nufrm)
        self.split_xy(nufrm)
    def xtr (self,stdiz=True,np = False):
        x = self.xtra
        if stdiz:
            x = procss(x)
        if np:
            x = pdcol2np(x)
        self.xtr = x
        return x
    def xte (self,stdiz=True,np = False):
        x=self.xtes
        if stdiz:
            x = procss(x)
        if np:
            x = pdcol2np(x)
        self.xte = x
        return x
    def yte (self,stdiz=True,np = True,yshuff = False):
        y = self.ytes
        y = y.reset_index(drop=True)
        if yshuff:
            y = myshuffle(y)
        if stdiz:
            y = procss(y).astype(bool)
            y=~y
            print ('dist of yte is:', y.sum()/y.shape[0])
        if np:
            y = pdcol2np(y)
        self.yte = y
        return y
    def ytr (self,stdiz=True,np = True):
        y = self.ytra
        if stdiz:
            y = procss(y).astype(bool)
            y = ~y
            print ('dist of ytr is:', y.sum()/y.shape[0])
        if np:
            y = pdcol2np(y)
        self.ytr = y
        return y
    def split_xy(self,f):
        y = f[[self.labels]]
        x = f.drop(self.labels, axis=1)
        self.xtra, self.xtes, self.ytra, self.ytes = train_test_split(x, y, test_size=args.trte_split, random_state=42)

