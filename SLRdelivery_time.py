# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:21:04 2024

@author: HP
"""

#Problem Statement: -
'''
    A logistics company recorded the time taken for delivery and the time taken for 
    the sorting of the items for delivery. Build a Simple Linear Regression model 
    to find the relationship between delivery time and sorting time with delivery 
    time as the target variable. Apply necessary transformations and record the RMSE 
    and correlation coefficient values for different models.
'''
#Business objective:
'''
    The objective is to find the relationship between delivery time and sorting
    time for a logistics company, with delivery time as the target variable.
'''
#Data Dictionary:
'''
    Delivery Time	=> Float	Target variable
    Sorting Time	=> Integer	Predictor variable
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
data = pd.read_csv('C:/0-Assignments/Assignments/SLReg/delivery_time.csv.xls')
data.shape
#(21, 2)
# Display data information
print("Data Information:")
print(data.info())
'''
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21 entries, 0 to 20
    Data columns (total 2 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Delivery Time  21 non-null     float64
     1   Sorting Time   21 non-null     int64  
    dtypes: float64(1), int64(1)
    memory usage: 468.0 bytes
    None
'''
# Summary statistics
print("\nSummary Statistics:")
print(data.describe())
'''
     Delivery Time  Sorting Time
    count      21.000000     21.000000
    mean       16.790952      6.190476
    std         5.074901      2.542028
    min         8.000000      2.000000
    25%        13.500000      4.000000
    50%        17.830000      6.000000
    75%        19.750000      8.000000
    max        29.000000     10.000000
'''

# Visualize the relationship between Delivery Time and Sorting Time
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Sorting Time', y='Delivery Time', data=data)
plt.title('Relationship between Delivery Time and Sorting Time')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.grid(True)
plt.show()

# Split data into X and y
X = data[['Sorting Time']]
y = data['Delivery Time']

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
print("RMSE:", rmse) #3.7478979383962705
print("R-squared:", r_squared) #-1.0207599586367362

# Plot the regression line along with the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Sorting Time', y='Delivery Time', data=data)
plt.title('Relationship between Delivery Time and Sorting Time')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')

# Plot the regression line
plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.grid(True)
plt.show()
