# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 04:09:24 2017

@author: Ali
"""
#### PART I DATA PREPROCESSING \

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #Preventing the Dummy Variable Trap 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### PART II MAKING THE ANN MODEL 
#Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#Initializing the ANN 
classifier = Sequential()
#Adding the input layer and the first hidden layer 
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu',input_dim = 11))
#Adding the second hidden layer 
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#Adding the output layer 
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) #if output = 3 then activation = softmax
#Compiling the ANN 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])
#Fitting the ANN to the Training Set 
classifier.fit(X_train, y_train, batch_size = 20, epochs =100 )

#PART III MAKING THE PREDICTIONS AND EVALUATING THE MODEL 
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #If y_pred larger than 0.5 then true, otherwise false 
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

