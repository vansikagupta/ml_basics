#Simple Linear Regression

#Data Preprocessing
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset and seperating independent and dependent variables
dataset = pd.read_csv('simple_linear_reg\Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#Spliting the dataset into test-train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#Most Simple linear regression model take care of feature scaling

