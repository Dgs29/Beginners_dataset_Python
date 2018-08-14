#Import library
import numpy as np
import pandas as pd

#Import dataset
df_train = pd.read_csv('train.csv')
df_train = df_train.drop(df_train.index[[61,829]])
df_test = pd.read_csv('test.csv')

#Splitting into training and test set
X_train = df_train.iloc[:,[2,4,5,6,7,9,11]].values
y_train = df_train.iloc[:,1].values
X_test = df_test.iloc[:,[1,3,4,5,6,8,10]].values

# Missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X_train[:,2:3])
X_train[:,2:3] = imputer.transform(X_train[:,2:3])

imputer_test1 = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer_test1 = imputer_test1.fit(X_test[:,2:3])
X_test[:,2:3] = imputer_test1.transform(X_test[:,2:3])

imputer2 = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer2 = imputer2.fit(X_test[:,5:6])
X_test[:,5:6] = imputer2.transform(X_test[:,5:6])

#Categorising values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_X_train = LabelEncoder()
X_train[:,1] = label_X_train.fit_transform(X_train[:,1])
X_train[:,6] = label_X_train.fit_transform(X_train[:,6])
onehot = OneHotEncoder(categorical_features = [6])
X_train = onehot.fit_transform(X_train).toarray()

label_X_test = LabelEncoder()
X_test[:,1] = label_X_test.fit_transform(X_test[:,1])
X_test[:,6] = label_X_test.fit_transform(X_test[:,6])
onehot_test = OneHotEncoder(categorical_features = [6])
X_test = onehot.fit_transform(X_test).toarray()

#Avoid dummy variable trap
X_train = X_train[:,1:]
X_test = X_test[:,1:]

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting SVM to the dataset
from sklearn.svm import SVC
classifier_SVM = SVC(kernel = 'rbf', C = 2, gamma = 0.1)
classifier_SVM.fit(X_train,y_train)

#Predicting test results
y_pred = classifier_SVM.predict(X_test)

#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_SVM, X = X_train, y = y_train, cv = 10)
SVM_accuracy = accuracies.mean()
accuracies.std()


#Fitting KNN to the test set
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors = 8,metric = 'minkowski', p = 2)
classifier_KNN.fit(X_train,y_train)
ypred_KNN = classifier_KNN.predict(X_test)

#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_SVM, X = X_train, y = y_train, cv = 10)
kfold_accuracy = accuracies.mean()
accuracies.std()

#Fitting NaiveBayes to the test set
from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(X_train,y_train)
ypred_NB = classifier_NB.predict(X_test)

#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_NB, X = X_train, y = y_train, cv = 10)
NB_accuracy = accuracies.mean()
accuracies.std()


#Fitting Decision Tree to the test set
from sklearn.tree import DecisionTreeClassifier
classifier_DecTree = DecisionTreeClassifier(criterion = 'entropy')
classifier_DecTree.fit(X_train,y_train)
ypred_DecTree = classifier_DecTree.predict(X_test)

#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_DecTree, X = X_train, y = y_train, cv = 10)
DTree_accuracy = accuracies.mean()
accuracies.std()


#Fitting RandomForest to the test set
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 6, criterion = 'entropy')
classifier_RF.fit(X_train,y_train)
ypred_RF = classifier_RF.predict(X_test)

#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_RF, X = X_train, y = y_train, cv = 10)
RF_accuracy = accuracies.mean()
accuracies.std()

#Grid Search
'''from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators':[5,6,7,8,9,10,11]}]
grid_search = GridSearchCV(estimator = classifier_RF,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_'''

