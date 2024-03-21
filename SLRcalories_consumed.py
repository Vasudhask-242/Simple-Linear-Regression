# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:45:41 2024

@author: HP
"""

#Probelm Statement
'''
    A certain food-based company conducted a survey with the help of a fitness company 
    to find the relationship between a person’s weight gain and the number of calories 
    they consumed in order to come up with diet plans for these individuals. 
    Build a Simple Linear Regression model with 
    calories consumed as the target variable. 
    Apply necessary transformations and record the RMSE and correlation coefficient 
    values for different models. 
'''

#Business Objective:
'''
    The objective is to find the relationship between a person's weight gain and 
    the number of calories they consumed to develop effective diet plans.
'''
#Data Dictionary:
'''    
    Weight gained (grams) =>	Integer	Target variable
    Calories Consumed	=> Integer	Predictor variable
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("C:/0-Assignments/Assignments/SLReg/calories_consumed.csv.xls")
data.shape
#(14, 2)

# Display data information
print(data.info())
'''
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14 entries, 0 to 13
    Data columns (total 2 columns):
     #   Column                 Non-Null Count  Dtype
    ---  ------                 --------------  -----
     0   Weight gained (grams)  14 non-null     int64
     1   Calories Consumed      14 non-null     int64
    dtypes: int64(2)
    memory usage: 356.0 bytes
    None
'''

# Summary statistics
print(data.describe())
'''
    Weight gained (grams)  Calories Consumed
    count              14.000000          14.000000
    mean              357.714286        2340.714286
    std               333.692495         752.109488
    min                62.000000        1400.000000
    25%               114.500000        1727.500000
    50%               200.000000        2250.000000
    75%               537.500000        2775.000000
    max              1100.000000        3900.000000
'''
# Visualize the relationship between Weight gained and Calories consumed
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Weight gained (grams)', y='Calories Consumed', data=data)
plt.title('Relationship between Weight gained and Calories consumed')
plt.xlabel('Weight gained (grams)')
plt.ylabel('Calories Consumed')
plt.grid(True)
plt.show()

# Split data into X and y
X = data[['Weight gained (grams)']]
y = data['Calories Consumed']

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
print("RMSE:", rmse)    #204.36123425476396
print("R-squared:", r_squared)  #0.9621097150609703

# Plot the regression line along with the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Weight gained (grams)', y='Calories Consumed', data=data)
plt.title('Relationship between Weight gained and Calories consumed')
plt.xlabel('Weight gained (grams)')
plt.ylabel('Calories Consumed')

# Plot the regression line
plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.grid(True)
plt.show()
