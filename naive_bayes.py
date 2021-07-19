# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:28:03 2021

@author: Sumukh
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('tennis.csv')
print(df.head(2))
    
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
df['outlook']=le.fit_transform(df['outlook'])
df['temp'] = le.fit_transform(df['temp'])
df['play'] = le.fit_transform(df['play'])
print (df['outlook'])

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)

from sklearn.naive_bayes import GaussianNB

class_1 = GaussianNB()
class_1.fit(x_train,y_train)

y_pred = class_1.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix

accuracy_score(y_pred,y_test)

cm = confusion_matrix(y_pred,y_test)

