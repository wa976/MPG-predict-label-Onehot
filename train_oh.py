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


data=pd.read_csv("./modeldata.csv")
# print(data.head())
#
# print(data.describe())
#
# print(data.info())

data = data.dropna()

# print(data.info())



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




print(data.head())
print(data.info())


x=data.drop(['MPG'],axis=1)
y=data['MPG']
xr,xt,yr,yt=train_test_split(x,y,test_size=0.1)

print(x)
print(y)




mod=LGBMRegressor(n_estimators=40)
model=make_pipeline(mod)
model.fit(x,y)
print(model)
kfold=KFold(n_splits=5)
score=cross_val_score(model,x,y,cv=kfold)
print(score)
yp=model.predict(xt)
print(r2_score(yt,yp))
print(mean_squared_error(yt,yp))
print(mean_squared_log_error(yt,yp))




joblib.dump(model, 'LGB_OH.pkl')


mod=RandomForestRegressor(n_estimators=100)
model=make_pipeline(mod)
print(model)
kfold=KFold(n_splits=5)
model.fit(x,y)
score=cross_val_score(model,x,y,cv=kfold)
print(score)
yp=model.predict(xt)
print(r2_score(yt,yp))
print(mean_squared_error(yt,yp))
print(mean_squared_log_error(yt,yp))


joblib.dump(model, 'RF_OH.pkl')


xgb1 = XGBRegressor()
parameters = {'n_estimators': [500]}
xgb_grid = GridSearchCV(xgb1,parameters,cv = 2)
xgb_grid.fit(x,y)
print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
yp=xgb_grid.predict(xt)
print(r2_score(yt,yp))
print(mean_squared_error(yt,yp))
print(mean_squared_log_error(yt,yp))


joblib.dump(xgb_grid, 'xgb_OH.pkl')

model = LinearRegression()
model.fit(x,y)
print(model)
kfold=KFold(n_splits=5)
model.fit(x,y)
score=cross_val_score(model,x,y,cv=kfold)
print(score)
yp=model.predict(xt)
print(r2_score(yt,yp))
print(mean_squared_error(yt,yp))
print(mean_squared_log_error(yt,yp))
joblib.dump(model, 'LR_OH.pkl')