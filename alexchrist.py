# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:49:16 2020

@author: 30697
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.isotonic import IsotonicRegression

df = pd.read_csv('flight_data.csv')
df = df.drop(['Unnamed: 18'], axis=1)
df = df.select_dtypes(include=[np.number])
df = df.dropna()

X_train, X_test, y_train, y_test = train_test_split(df.drop(['DEP_DEL15'], axis=1), df['DEP_DEL15'], test_size=0.25, random_state=0)

my_models = ["GaussianNB()", "make_pipeline(StandardScaler(), SVC(gamma='auto'))", "GradientBoostingClassifier(random_state=0)", "RandomForestClassifier(max_depth=2, random_state=0)"]


gnb = eval(my_models[0])
y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
#print("Naive Bayes --- Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != y_pred_gnb).sum()))
print("Naive Bayes --- Accuracy : ", (X_test.shape[0] - (y_test != y_pred_gnb).sum() ) / X_test.shape[0] )


clf = eval(my_models[1])
y_pred_svm = clf.fit(X_train, y_train).predict(X_test)
#print("Support Vector Machine --- Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != y_pred_svm).sum()))
print("Support Vector Machine --- Accuracy : ", (X_test.shape[0] - (y_test != y_pred_svm).sum() ) / X_test.shape[0] )


clf = eval(my_models[2])
y_pred_gbt = clf.fit(X_train, y_train).predict(X_test)
#print("Gradient Boosting Tree --- Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != y_pred_gbt).sum()))
print("Gradient Boosting Tree --- Accuracy : ", (X_test.shape[0] - (y_test != y_pred_gbt).sum() ) / X_test.shape[0] )

clf = eval(my_models[3])
y_pred_rf = clf.fit(X_train, y_train).predict(X_test)
#print("Random Forest Classification --- Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != y_pred_rf).sum()))
print("Random Forest Classification --- Accuracy : ", (X_test.shape[0] - (y_test != y_pred_rf).sum() ) / X_test.shape[0] )


my_reg_models = ["linear_model.LinearRegression()", "IsotonicRegression()"]

reg = eval(my_reg_models[0])
y_pred_reg = reg.fit(X_train, y_train).predict(X_test)
print('Linear Regression --- Mean squared error: %.2f'   % mean_squared_error(y_test, y_pred_reg))

iso_reg = eval(my_reg_models[1])
y_pred_iso = iso_reg.fit(list(X_train['DISTANCE'].values), y_train).predict(list(X_test['DISTANCE'].values))
print('Isotonic Regression --- Mean squared error: %.2f'   % mean_squared_error(y_test, y_pred_iso))
