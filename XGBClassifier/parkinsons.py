import numpy as np
import pandas as pd
import os,sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## Read Data in
df = pd.read_csv('parkinsons.data')
print(df.head())

## Read all rows with columns as specified
features = df.loc[:,df.columns!='status'].values[:,1:]
labels = df.loc[:,'status'].values

##Print number of rows with given specification
print(labels[labels==1].shape[0],labels[labels==0].shape[0])

##Scale features to bewteen -1 and 1
scaler = MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels

##Create training and testing split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=7)

## Train the model
model = XGBClassifier()
model.fit(x_train,y_train)

##Calculate accruacy
y_pred = model.predict(x_test)
print(accuracy_score(y_test,y_pred)*100)
