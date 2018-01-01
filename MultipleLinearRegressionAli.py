# Multiple Linear Regression 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X= X[:, 1:] # Library takes care of it here so we don't need to do that 


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Regression to the Training Set 
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set Results 
y_pred= regressor.predict(X_test)

#Building the Optimal model Using Backward ELimination 
###Stats model library 
import statsmodels.formula.api as sm
#### Doesn't take in account the "b0" 
#### Adding the column of ones could help 
###X = np.append(arr= X, values= np.ones((50, 1)).astype(int), axis=1) THIS adds ones to the matrix X to the end but we want it in the starting
X = np.append(arr= np.ones((50, 1)).astype(int), values= X, axis=1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS= sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS= sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS= sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS= sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS= sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS= sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()