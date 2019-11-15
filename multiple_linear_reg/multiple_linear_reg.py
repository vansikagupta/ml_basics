#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset and seperating independent and dependent variables
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


#Handling categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:, 3] = le.fit_transform(X[:, 3])
#This will add attribute of order to the categorical feature, which is incorrect for this dataset
#Using OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


#Handling Dummy Variable trap
X = X[:, 1:]


#Spliting the dataset into test-train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


#Creating regressor and fitting train data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#making predictions on test data
y_pred = regressor.predict(X_test)


#Building the optimal model using Backward Elimination
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((X.shape[0],1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#x2 has the highest p-value which is greater then the significance level 0.05, thus remove x2 and repeat steps
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#x1 has the highest p-value which is greater then the significance level 0.05, thus remove x2 and repeat steps
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#x2 in X_opt (4th column in X)  has the highest p-value which is greater then the significance level 0.05, thus remove x2 and repeat steps
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#x2 (5th column)  has the highest p-value which is greater then the significance level 0.05, thus remove x2 and repeat steps
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#Now all independent variables have p-value less than 0.05, so our optimal team of independent variables is now ready
