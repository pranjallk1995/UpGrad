import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv("Bank.csv")
dataset = dataset.drop('day', 1)

X = pd.DataFrame(dataset.iloc[:, :-1])
Y = pd.DataFrame(dataset.iloc[:, 15])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
classifier_encoder = LabelEncoder()
Y["y"] = classifier_encoder.fit_transform(Y["y"])
X["housing"] = classifier_encoder.fit_transform(X["housing"])
X["loan"] = classifier_encoder.fit_transform(X["loan"])
X["default"] = classifier_encoder.fit_transform(X["default"])

X = pd.get_dummies(X, columns=['job'], prefix = ['job'])
X = pd.get_dummies(X, columns=['marital'], prefix = ['marital'])
X = pd.get_dummies(X, columns=['education'], prefix = ['education'])
X = pd.get_dummies(X, columns=['contact'], prefix = ['contact'])
X = pd.get_dummies(X, columns=['month'], prefix = ['month'])
X = pd.get_dummies(X, columns=['poutcome'], prefix = ['poutcome'])

#performing feature scaling
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X = X.astype('float')
X = pd.DataFrame(X_sc.fit_transform(X))

#splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size = 0.25)

#fitting SVM classifier
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, gamma = 'auto')
classifier.fit(X_train, Y_train.values.ravel())

#predicting values
Y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred))