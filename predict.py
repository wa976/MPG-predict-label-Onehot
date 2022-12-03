import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_squared_log_error,make_scorer
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression
import pickle

data=pd.read_csv("./newinput.csv")
print(data.head())

print(data.info())


lab = LabelEncoder()
pkl_file = open('Label_encoder_1.pkl', 'rb')
lab = pickle.load(pkl_file)
pkl_file.close()
print(lab.classes_)
data['cyl4'] = lab.transform(data['cyl4'])

pkl_file = open('Label_encoder_2.pkl', 'rb')
lab = pickle.load(pkl_file)
pkl_file.close()
print(lab.classes_)
data['Mfg'] = lab.transform(data['Mfg'])

pkl_file = open('Label_encoder_3.pkl', 'rb')
lab = pickle.load(pkl_file)
pkl_file.close()
print(lab.classes_)
data['Origin'] = lab.transform(data['Origin'])

pkl_file = open('Label_encoder_4.pkl', 'rb')
lab = pickle.load(pkl_file)
pkl_file.close()
print(lab.classes_)
data['when'] = lab.transform(data['when'])




print(data.info())
print(data.head())

print(data)

x=data

print(x)



mod=LGBMRegressor(n_estimators=40)
model=make_pipeline(mod)
model = joblib.load('LGB.pkl')
y_pred = model.predict(x)
print(y_pred)

mod=RandomForestRegressor(n_estimators=100)
model=make_pipeline(mod)
model = joblib.load('RF.pkl')
y_pred = model.predict(x)
print(y_pred)

xgb1 = XGBRegressor()
parameters = {'n_estimators': [500]}
model = GridSearchCV(xgb1,parameters,cv = 2)
model = joblib.load('xgb.pkl')
y_pred = model.predict(x)
print(y_pred)

model = LinearRegression()
model = joblib.load('LR.pkl')
y_pred = model.predict(x)
print(y_pred)