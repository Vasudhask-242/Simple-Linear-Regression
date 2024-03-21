# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:38:49 2024

@author: HP
"""
#Problem Statement: -
'''
    A certain organization wants an early estimate of their employee churn out rate. 
    So the HR department gathered the data regarding the employee’s salary hike and
    the churn out rate in a financial year. The analytics team will have to perform
    an analysis and predict an estimate of employee churn based on the salary hike. 
    Build a Simple Linear Regression model with churn out rate as the target variable. 
    Apply necessary transformations and record the RMSE and correlation coefficient 
    values for different models.
'''
#Business Problem:
'''
    The objective is to predict the employee churn out rate based on the salary 
    hike.
'''
#Data Dictionary:
'''
    Salary_hike	Integer	Predictor variable
    Churn_out_rate	Integer	Target variable
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
data = pd.read_csv('C:/0-Assignments/Assignments/SLReg/emp_data.csv.xls')
data.shape
#(10, 2)
data.info()
'''
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 2 columns):
     #   Column          Non-Null Count  Dtype
    ---  ------          --------------  -----
     0   Salary_hike     10 non-null     int64
     1   Churn_out_rate  10 non-null     int64
    dtypes: int64(2)
    memory usage: 292.0 bytes
'''

data.describe()
'''
           Salary_hike  Churn_out_rate
    count    10.000000       10.000000
    mean   1688.600000       72.900000
    std      92.096809       10.257247
    min    1580.000000       60.000000
    25%    1617.500000       65.750000
    50%    1675.000000       71.000000
    75%    1724.000000       78.750000
    max    1870.000000       92.000000
'''

# Visualize the relationship between Salary hike and Churn out rate
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Salary_hike', y='Churn_out_rate', data=data)
plt.title('Relationship between Salary hike and Churn out rate')
plt.xlabel('Salary hike')
plt.ylabel('Churn out rate')
plt.grid(True)
plt.show()

# Split data into X and y
X = data[['Salary_hike']]
y = data['Churn_out_rate']

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
print("RMSE:", rmse) #2.7570866858265246
print("R-squared:", r_squared) #0.9425215350233506

# Plot the regression line along with the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Salary_hike', y='Churn_out_rate', data=data)
plt.title('Relationship between Salary hike and Churn out rate')
plt.xlabel('Salary hike')
plt.ylabel('Churn out rate')

# Plot the regression line
plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.grid(True)
plt.show()

