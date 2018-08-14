# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Import libraries
import numpy as np
import pandas as pd

#Import dataset
df_train = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
df_test = pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')

#splitting dataset into training and test set
X_train = df_train.iloc[:,1:-1].values
y_train = df_train.iloc[:,12].values
X_test = df_test.iloc[:,1:].values

#Missing values
#--------------training set---------
from sklearn_pandas import CategoricalImputer
imputer_train_cat = CategoricalImputer()
imputer_train_cat = imputer_train_cat.fit(X_train[:,[0,1,4]])
X_train[:,[0,1,4]] = imputer_train_cat.transform(X_train[:,[0,1,4]])

for i in range(0,614):
    if X_train[:,2][i] == '3+':
        X_train[:,2][i] = 3
    else:
        continue
       
from sklearn.preprocessing import Imputer
imputer_train_num = Imputer(missing_values = "NaN",strategy = "mean", axis = 0)
imputer_train_num = imputer_train_num.fit(X_train[:,[2,7,8,9]])
X_train[:,[2,7,8,9]] = imputer_train_num.transform(X_train[:,[2,7,8,9]])

#--------------test set------------------
imputer_test_cat = CategoricalImputer()
imputer_test_cat = imputer_test_cat.fit(X_test[:,[0,4]])
X_test[:,[0,4]] = imputer_train_cat.transform(X_test[:,[0,4]])

for i in range(0,367):
    if X_test[:,2][i] == '3+':
        X_test[:,2][i] = 3
    else:
        continue
       
from sklearn.preprocessing import Imputer
imputer_test_num = Imputer(missing_values = "NaN",strategy = "mean", axis = 0)
imputer_train_num = imputer_test_num.fit(X_test[:,[2,7,8,9]])
X_test[:,[2,7,8,9]] = imputer_test_num.transform(X_test[:,[2,7,8,9]])

#Categorical Split
#------------ training set---------------
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_X1 = LabelEncoder()
X_train[:,0] = label_X1.fit_transform(X_train[:,0])
X_train[:,1] = label_X1.fit_transform(X_train[:,1])
X_train[:,3] = label_X1.fit_transform(X_train[:,3])
X_train[:,4] = label_X1.fit_transform(X_train[:,4])
X_train[:,10] = label_X1.fit_transform(X_train[:,10])

onehot = OneHotEncoder(categorical_features = [10])
X_train = onehot.fit_transform(X_train).toarray()

y_train = label_X1.fit_transform(y_train)

#------------- test set--------------------
label_X2 = LabelEncoder()
X_test[:,0] = label_X2.fit_transform(X_test[:,0])
X_test[:,1] = label_X2.fit_transform(X_test[:,1])
X_test[:,3] = label_X2.fit_transform(X_test[:,3])
X_test[:,4] = label_X2.fit_transform(X_test[:,4])
X_test[:,10] = label_X2.fit_transform(X_test[:,10])

onehot = OneHotEncoder(categorical_features = [10])
X_test = onehot.fit_transform(X_test).toarray()

#Removing dummy variable trap
X_train = X_train[:,1:]
X_test = X_test[:,1:]

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting SVM to the dataset
from sklearn.svm import SVC
classifier_SVM = SVC(kernel = 'linear', C = 1)
classifier_SVM.fit(X_train,y_train)

accuracy_SVM = round(classifier_SVM.score(X_train,y_train)*100,2)

#Fitting Logistic Regression to the dataset
from sklearn.linear_model import LogisticRegression
classifier_log = LogisticRegression()
classifier_log.fit(X_train,y_train)
accuracy_Log = round(classifier_log.score(X_train,y_train)*100,2)

#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_SVM, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,100,1000],'kernel':['linear']}],
        
grid_search = GridSearchCV(estimator = classifier_SVM,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

grid_search = GridSearchCV()
#Building model using Backward Elimination
'''import statsmodels.formula.api as sm
X_train = np.append(arr = np.ones((614,1)).astype(int),values = X_train, axis = 1)

X_opt = X_train[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,1,2,3,4,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,1,2,4,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,1,2,4,5,6,9,10,11]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,1,2,4,6,9,10,11]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,1,2,4,6,9,10,11]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,1,2,4,6,9,10]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,1,4,6,9,10]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,1,4,6,9]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()'''


