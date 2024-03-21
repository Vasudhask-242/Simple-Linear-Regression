# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:47:20 2024

@author: HP
"""
#Problem Statement: -
'''
    The head of HR of a certain organization wants to automate their salary hike 
    estimation. The organization consulted an analytics service provider and asked 
    them to build a basic prediction model by providing them with a dataset that 
    contains the data about the number of years of experience and the salary hike 
    given accordingly. Build a Simple Linear Regression model with salary as the 
    target variable. Apply necessary transformations and record the RMSE and 
    correlation coefficient values for different models.
'''
#Business objective:
'''
    The objective is to automate salary hike estimation based on the number of 
    years of experience.
'''
#Data Dictionary:
'''
    YearsExperience =>	Float	Predictor variable
    Salary	        =>  Float	Target variable
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('C:/0-Assignments/Assignments/SLReg/Salary_Data.csv.xls')
data.shape
# (30, 2)
data.info()
'''
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30 entries, 0 to 29
    Data columns (total 2 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   YearsExperience  30 non-null     float64
     1   Salary           30 non-null     float64
    dtypes: float64(2)
    memory usage: 612.0 bytes
'''

data.describe()
'''
            YearsExperience         Salary
    count        30.000000      30.000000
    mean          5.313333   76003.000000
    std           2.837888   27414.429785
    min           1.100000   37731.000000
    25%           3.200000   56720.750000
    50%           4.700000   65237.000000
    75%           7.700000  100544.750000
    max          10.500000  122391.000000
'''

# Visualize the relationship between Years of Experience and Salary
plt.figure(figsize=(10, 6))
sns.scatterplot(x='YearsExperience', y='Salary', data=data)
plt.title('Relationship between Years of Experience and Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.grid(True)
plt.show()

# Split data into X and y
X = data[['YearsExperience']]
y = data['Salary']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculate R-squared
r_squared = r2_score(y_test, y_pred)

# Print model evaluation metrics
print("\nModel Evaluation Metrics:")
print("RMSE:", rmse) #7059.04362190151
print("R-squared:", r_squared) #0.9024461774180497

# Plot the regression line along with the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='YearsExperience', y='Salary', data=data)
plt.title('Relationship between Years of Experience and Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

# Plot the regression line
plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.grid(True)
plt.show()

