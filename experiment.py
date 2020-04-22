import pandas as pd
import numpy as np

list = ['rmse','rmsle']
mydict=dict()
for item in list:
    mydict[item]={'trcol':'train-'+item+'-mean',
                  'tecol':'test-'+item+' - mean',
                  'ylab': item,
                  'title': item}

print (mydict.items())

'''

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
