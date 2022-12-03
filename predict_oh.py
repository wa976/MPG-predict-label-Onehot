import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_squared_log_error,make_scorer
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression
import pickle



data_train=pd.read_csv("./modeldata.csv")
data_train = data_train.dropna()

data_train['MPG'] = 'train'

print(data_train)

data_new=pd.read_csv("./newinput.csv")
data_new['MPG'] = 'score'

print(data_new)

data = pd.concat([data_train,data_new])

print(data)



add_colums = pd.get_dummies(data['cyl4'])
print(add_colums)
data.drop(['cyl4'],axis=1,inplace=True)
print(data.columns)
data = data.join(add_colums)
print(data)


add_colums = pd.get_dummies(data['Mfg'])
print(add_colums)
data.drop(['Mfg'],axis=1,inplace=True)
print(data.columns)
data = data.join(add_colums)
print(data)


add_colums = pd.get_dummies(data['Origin'])
print(add_colums)
data.drop(['Origin'],axis=1,inplace=True)
print(data.columns)
data = data.join(add_colums)
print(data)

add_colums = pd.get_dummies(data['when'])
print(add_colums)
data.drop(['when'],axis=1,inplace=True)
print(data.columns)
data = data.join(add_colums)
print(data)



print(data.info())
print(data.head())

print(data)

data_train = data[data['MPG'] == 'train']
data_new = data[data['MPG'] == 'score']

data_new = data_new.drop('MPG', axis=1)

x=data_new

print(x)



mod=LGBMRegressor(n_estimators=40)
model=make_pipeline(mod)
model = joblib.load('LGB_OH.pkl')
y_pred = model.predict(x)
print(y_pred)

mod=RandomForestRegressor(n_estimators=100)
model=make_pipeline(mod)
model = joblib.load('RF_OH.pkl')
y_pred = model.predict(x)
print(y_pred)

xgb1 = XGBRegressor()
parameters = {'n_estimators': [500]}
model = GridSearchCV(xgb1,parameters,cv = 2)
model = joblib.load('xgb_OH.pkl')
y_pred = model.predict(x)
print(y_pred)

model = LinearRegression()
model = joblib.load('LR_OH.pkl')
y_pred = model.predict(x)
print(y_pred)