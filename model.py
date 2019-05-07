# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 08:41:55 2018

@author: lenovo
"""

# Titanic Data set # Data Analysis
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

titanic = pd.read_csv('Model/titanic_train.csv')
include = ['Age', 'Sex', 'Embarked', 'Survived']

titanic_df = titanic[include]
titanic_df.dropna(inplace = True)
X = titanic_df.drop(['Survived'], axis = 1)
y = titanic_df.Survived
imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
nom = LabelEncoder()
X['Sex'] = nom.fit_transform(X['Sex'])
X['Embarked'] = nom.fit_transform(X['Embarked'])

print(type(X))
lr = LogisticRegression()
lr.fit(X,y)


joblib.dump(lr, 'logit_model.pkl')

