# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 00:37:18 2017

@author: Ali
"""

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
dataset = pd.read_csv('bank.csv', sep = ";")
X = (dataset.iloc[:, 0:16].values)
y = (dataset.iloc[:, 16].values)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])
labelencoder_X_6 = LabelEncoder()
X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])
labelencoder_X_7 = LabelEncoder()
X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7])
labelencoder_X_8 = LabelEncoder()
X[:, 8] = labelencoder_X_8.fit_transform(X[:, 8])
labelencoder_X_10 = LabelEncoder()
X[:, 10] = labelencoder_X_10.fit_transform(X[:, 10])
labelencoder_X_15 = LabelEncoder()
X[:, 15] = labelencoder_X_15.fit_transform(X[:, 15])
onehotencoder = OneHotEncoder(categorical_features = [1,2,3,4,6,7,10,15])
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

#==============================================================================
# ### PART II MAKING THE ANN MODEL 
# #Importing the Keras libraries and packages
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# #Initializing the ANN 
# classifier = Sequential()
# #Adding the input layer and the first hidden layer 
# classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu',input_dim = 11))
# #Adding the second hidden layer 
# classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# #Adding the output layer 
# classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) #if output = 3 then activation = softmax
# #Compiling the ANN 
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])
# #Fitting the ANN to the Training Set 
# classifier.fit(X_train, y_train, batch_size = 20, epochs =100 )
# 
# #PART III MAKING THE PREDICTIONS AND EVALUATING THE MODEL 
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5) #If y_pred larger than 0.5 then true, otherwise false 
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
#==============================================================================
###### PART 1 USING THE SVM ALGORITHM 
# Fitting Kernel SVM to the Training set
#from sklearn.svm import SVC 
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(X_train, y_train)
##### 88.79% accurate 

#### PART 2 USING THE NAIVE BAYES 
#from sklearn.naive_bayes import GaussianNB 
#classifier = GaussianNB()
#classifier.fit(X_train,y_train)
### 82.8% accurate 

#### PART 3 Decision Tree Classification 
# Fitting classifier to the Training set
#from sklearn.tree import DecisionTreeClassifier 
#classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
#classifier.fit(X_train, y_train)
### 88.06% accurate 

### PART 4 Random Forest Classification 
# Fitting Random Forest Classification to the Training set
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state= 0)
#classifier.fit(X_train, y_train)
#### 89.17 accurate 

#PART 5 K-NN 
# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p= 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

