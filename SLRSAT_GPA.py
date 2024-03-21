# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:58:55 2024

@author: HP
"""

#Problem Statement: -
'''
    A certain university wants to understand the relationship between students’ 
    SAT scores and their GPA. Build a Simple Linear Regression model with GPA as 
    the target variable and record the RMSE and correlation coefficient values for 
    different models.
'''

#Business objective:
'''
The objective is to understand the relationship between students' SAT scores 
and their GPA.
'''
#Data Dictionary:
'''
    SAT_Scores	Integer	Predictor variable
    GPA	Float	Target variable
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
data = pd.read_csv('C:/0-Assignments/Assignments/SLReg/SAT_GPA.csv.xls')
data.shape
# (200, 2)
data.info()
'''
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200 entries, 0 to 199
    Data columns (total 2 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   SAT_Scores  200 non-null    int64  
     1   GPA         200 non-null    float64
    dtypes: float64(1), int64(1)
    memory usage: 3.3 KB
'''

data.describe()
'''
           SAT_Scores         GPA
    count  200.000000  200.000000
    mean   491.810000    2.849500
    std    174.893834    0.541076
    min    202.000000    2.000000
    25%    349.750000    2.400000
    50%    480.500000    2.800000
    75%    641.500000    3.400000
    max    797.000000    3.900000
'''


# Visualize the relationship between SAT Scores and GPA
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SAT_Scores', y='GPA', data=data)
plt.title('Relationship between SAT Scores and GPA')
plt.xlabel('SAT Scores')
plt.ylabel('GPA')
plt.grid(True)
plt.show()

# Split data into X and y
X = data[['SAT_Scores']]
y = data['GPA']

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
print("RMSE:", rmse)    #0.547475367289355
print("R-squared:", r_squared)  #0.0057906035703984005

# Plot the regression line along with the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SAT_Scores', y='GPA', data=data)
plt.title('Relationship between SAT Scores and GPA')
plt.xlabel('SAT Scores')
plt.ylabel('GPA')

# Plot the regression line
plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.grid(True)
plt.show()

