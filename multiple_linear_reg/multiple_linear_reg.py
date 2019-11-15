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