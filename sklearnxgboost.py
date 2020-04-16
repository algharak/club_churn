import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error,accuracy_score,classification_report
####setup parameters
tr_file       = './dataset/club_churn_train_std.csv'
te_file       ='./dataset/club_churn_test_std.csv'


pd_tr =pd.read_csv(tr_file)
pd_te =pd.read_csv(te_file)
tr_cols = list(pd_tr.columns.values)
te_cols = list(pd_te.columns.values)
Y_tr = pd_tr [tr_cols[0]]

#Y_tr = np.random.choice([1,0], 9325)   #model too good.  wanted to see if real
Y_tr = np.array(Y_tr, dtype='int32')
print (Y_tr.shape[0], type (Y_tr.shape))
X_tr = pd_tr [tr_cols[1:]]
X_tr = np.array(X_tr, dtype = 'float32')

ycol = te_cols[0]
xcol = te_cols [1:]

Y_te = pd_te [te_cols[0]]
Y_te = np.array(Y_te, dtype='int32')
X_te = pd_te [te_cols[1:]]
X_te = np.array(X_te, dtype = 'float32')

xgb_model = xgb.XGBClassifier().fit(X_tr, Y_tr)
Y_pred = xgb_model.predict(X_te)
print('mse is :  ', mean_squared_error(Y_te, Y_pred))
print('accuracy is :  ', accuracy_score(Y_te, Y_pred))
print('report:  ')
print(classification_report(Y_te, Y_pred))

clf = GridSearchCV(xgb_model,
                   {'max_depth': [4,6,8],
                    'n_estimators': [12,25,50],
                    'eta':[0.01,0.02,0.05]}, verbose=0)

clf.fit(X_tr, Y_tr)
print(clf.best_score_)
print(clf.best_params_)



'''
print("Zeros and Ones from the Digits dataset: binary classification")
digits = load_digits(2)
y = digits['target']
X = digits['data']
kf = KFold(n_splits=2, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(confusion_matrix(actuals, predictions))

print("Iris: multiclass classification")
iris = load_iris()
y = iris['target']
X = iris['data']
kf = KFold(n_splits=2, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(confusion_matrix(actuals, predictions))

print("Boston Housing: regression")
boston = load_boston()
y = boston['target']
X = boston['data']
kf = KFold(n_splits=2, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBRegressor().fit(X[train_index], y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(mean_squared_error(actuals, predictions))

print("Parameter optimization")
y = boston['target']
X = boston['data']
xgb_model = xgb.XGBRegressor()
clf = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
clf.fit(X,y)
print(clf.best_score_)
print(clf.best_params_)

# The sklearn API models are picklable
print("Pickling sklearn API models")
# must open in binary format to pickle
pickle.dump(clf, open("best_boston.pkl", "wb"))
clf2 = pickle.load(open("best_boston.pkl", "rb"))
print(np.allclose(clf.predict(X), clf2.predict(X)))

# Early-stopping

X = digits['data']
y = digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc",
        eval_set=[(X_test, y_test)])

'''