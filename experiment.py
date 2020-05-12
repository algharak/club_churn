import pandas as pd
import numpy as np
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

'''

class obj_ ():
    def __init__(self,a,b):
        self.a = a
        self.b = b
    #@classmethod
    def ax_optim(self,n):
        #full_param = {**par, **args.base_param}
        #mod = xgb_kl(**full_param)
        #kfold = StratifiedShuffleSplit(n_splits=4)
        #cv_results = cross_val_score(mod, cls.xtr, cls.yte, cv=kfold, scoring='recall')
        #loss = 1 - max(cv_results)
        return self.a + self.b - n
    def ax2 (self,j):
        return self.ax_optim(j)

xz = obj_(4,5)
print (xz.ax_optim(99))
qqq = xz.ax2
print (qqq(2))
def xtr (d):
    return d.xtr()


list = ['rmse','rmsle']
mydict=dict()
for item in list:
    mydict[item]={'trcol':'train-'+item+'-mean',
                  'tecol':'test-'+item+' - mean',
                  'ylab': item,
                  'title': item}

print (mydict.items())
  x = procss(fr.drop(labels, axis=1))
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
le = LabelEncoder()

def double(x):
    return 2*x

a = [double,1]

print(a[0](5))
exit()

dict={'here':double}
print (dict.keys())
print (dict.values())

print(dict['here'](2))

exit()

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)
df.drop(columns=['A'],inplace=True)
print (df)
exit()
print(type(df['A']))
coladf = df[['A']]
print(type(df[['A']]))
df.loc[:,'A'] = 18
print  (df)
df['A']=df['B']
print (type(df['A']))
print (df.loc[:,'A'])
print (type(df.loc[:,'A']))
#a = np.array([[1],[2],[3],[4],[4],[5]])
a = np.array(range(6))
print(a)
df.loc[:,'A'] = a
print(df.info())

exit()

scaler = MinMaxScaler(feature_range=(0, 1))

dtext = ['20160101']
tt = dtext[0]
yr = int(tt [:4])
mo = int (tt[4:6])
day = int (tt[6:])

def func(a):
    yr = int(a[:4])
    mo = int(a[4:6])
    day = int(a[6:])
    return yr*365+mo*12+day





df = pd.DataFrame({'col':['11921123']})
print(df.head())
#df[['A']].apply(lambda x: func(df['A']))
df=df['col'].apply(func)
print(df.head())

df = pd.DataFrame({'date': ['19910101','19921204','19930104']})
dates = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
df.info()

print (df)
datesnp=dates.values
datesnp=datesnp.reshape(len(datesnp), 1)

scaler = scaler.fit(datesnp)
normalized = scaler.transform(datesnp)

#(aaa.time).total_seconds()
#print(aaa)
exit()


mf = pd.DataFrame({'col1':[1,2,1,3,4,2,12,33,33]})
mfn = mf.to_numpy()
colname = mf.columns
mm.fit(mf[colname])
#mm.fit(mf[['col1']])
mf[colname]=mm.transform(mf[colname])
print ('hello')
exit()





#le.fit(mf)
colname=mf.columns[0]
le.fit(mf[colname])
#le.fit(mf['col1'])
print (le.classes_)
tolist = mf['col1'].tolist()
#tolist = mf['col1'].tolist()
print (type(le.transform(tolist)))
print (le.transform(tolist))
mf['col1'] = le.transform(tolist)
print (mf)

print ('hello')


print ('print (myframe)  ', myframe)
print('-----------------------------------------------------')

myframe['col1'] = le.fit(myframe['col1'])
print ('myframe.columns[0]')
print (myframe.columns[0])
print('-----------------------------------------------------')

print ('(type(myframe.columns[0]))')
print (type(myframe.columns[0]))
print('-----------------------------------------------------')

print ('le.classes')
print (le.classes_)
print('-----------------------------------------------------')

print ('type(myframe)')
print (type(myframe))
print('-----------------------------------------------------')
print ('myframe.shape')
print (myframe.shape)
print('-----------------------------------------------------')
print ('myframe.info')
print (myframe.info)
print('-----------------------------------------------------')
print('le.transform([1,4,33])')
print(le.transform([1,4,33]))
print('-----------------------------------------------------')
print('type((myframe.loc[:,[col1]].to_numpy()))')
print(type((myframe.loc[:,['col1']].to_numpy())))
print('-----------------------------------------------------')
#aaa= myframe.loc[:,['col1']].to_numpy()

ccc = myframe.loc[:,['col1']].to_numpy()
print ('ccc = myframe.loc[:,[col1]].to_numpy()')

xxx = myframe.loc[:,'col1'].to_numpy()

print (xxx,'  ', 'xxx is')

fff = list(myframe.loc[:,['col1']])
aaa = myframe['col1']
bbb = myframe.loc[:,'col1']
ddd =  list (ccc)
eee = [1,2,33]
print (type(aaa))
print (type(bbb))
print (type(ccc))
print (fff)
print (type(fff))



print (le.transform(fff))
#myframe.loc[:,['col']] = le.transform(myframe['col1'].to_numpy())
'''


#print (myframe.head())
