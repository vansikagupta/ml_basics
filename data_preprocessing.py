#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset and seperating independent and dependent variables
dataset = pd.read_csv('Documents\Machine_Learning_Udemy\ml_basics\Data.csv')
X = dataset.iloc[:,:3]
y = dataset.iloc[:,3]


#Handling missing values
#sklearn.preprocessing.Imputer is now deprecated
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])


#Handling categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x = LabelEncoder()
X.iloc[:, 0] = le.fit_transform(X.iloc[:, 0])
#This will add attribute of order to the categorical feature, which is incorrect for this dataset
#Using OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#Encoding dependent variable
le_y = LabelEncoder()
y = le_y.fit_transform(y)


#Spliting the dataset into test-train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[:, 3:5] = scaler.fit_transform(X_train[:, 3:5])
X_test[:, 3:5] = scaler.transform(X_test[:, 3:5])
#Dependent variables are scaled already for this dataset