# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:09:00 2018

@author: Sanjeevi
"""
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:,1:5].values
y = dataset.iloc[:,5].values

#Categorical split
from sklearn.preprocessing import LabelEncoder
label_y = LabelEncoder()
y = label_y.fit_transform(y)

#Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3,random_state = 0)

#Fitting SVM to the dataset
from sklearn.svm import SVC
classifier_SVM = SVC(C = 6,kernel = 'rbf',gamma = 0.2)
classifier_SVM.fit(X_train,y_train)
ypred_SVM = classifier_SVM.predict(X_test)
#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_SVM, X = X_train, y = y_train, cv = 10)
SVM_accuracy = accuracies.mean()

#Fitting KNN to the dataset
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors = 7,metric = 'minkowski', p =2)
classifier_KNN.fit(X_train,y_train)
ypred_KNN = classifier_KNN.predict(X_test)
#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_KNN, X = X_train, y = y_train, cv = 10)
Kfold_accuracy = accuracies.mean()

#Fitting NaiveBayes to the dataset
from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(X_train,y_train)
ypred_NB = classifier_NB.predict(X_test)
#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_NB, X = X_train, y = y_train, cv = 10)
NB_accuracy = accuracies.mean()

#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm_SVM = confusion_matrix(y_test,ypred_SVM)

#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm_KNN = confusion_matrix(y_test,ypred_KNN)

#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm_NB = confusion_matrix(y_test,ypred_NB)