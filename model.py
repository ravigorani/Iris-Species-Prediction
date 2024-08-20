import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

data=pd.read_csv("data/iris.csv")

x=data.iloc[:,:-1]
y=data["species"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

lr = LogisticRegression()
lr.fit(x_train,y_train)

pickle.dump(lr,open("model.pkl","wb"))