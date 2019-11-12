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


#Creating the Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#fit the training data
regressor.fit(X_train, y_train) 
#predict the test data
y_pred = regressor.predict(X_test) 
#compare prediction and observation
#Plotting Training data set
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Training data plot')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#Plotting Test data set 
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Test data plot')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()