import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from flask import Flask, render_template,request
import pickle
#data = pd.read_csv ('/content/drive/MyDrive/data')
data = pd.read_csv("newdata.csv")
#from google.colab import drive
#drive.mount('/content/drive')

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data.Account=encoder.fit_transform(data.Account)

# Feature Selection
data.info()
#data.astype(int)
data.info()
#Spliting into X and Y
X=data.drop('impact',axis=1)
y=data['impact']
X
y
#Spliting into Training and testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)# training Model
pred=model.predict(X_test) #predicting 'impact'
# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

from sklearn.metrics import (accuracy_score,confusion_matrix)
accuracy_score(y_test,pred)
