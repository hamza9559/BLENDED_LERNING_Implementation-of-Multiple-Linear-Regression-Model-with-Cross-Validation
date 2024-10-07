# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Load the dataset and define features (Age, Mileage, Horsepower) and target (Price).
2. Cross-Validation & Prediction: Use K-Fold Cross-Validation to train a Multiple Linear Regression model and predict prices for each fold.
3. Evaluation: Calculate the Mean Squared Error (MSE) to assess the model's accuracy in predicting car prices.
4. Visualization: Plot the actual vs predicted car prices to visually compare how well the model predicts the prices.

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: HAMZA FAROOQUE 
RegisterNumber:  212223040054
*/

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data - car attributes (Age, Mileage, Horsepower) and Price
data = {
    'Age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Mileage': [5000, 15000, 25000, 35000, 45000, 55000, 65000, 75000, 85000, 95000],
    'Horsepower': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    'Price': [20000, 18500, 17500, 16500, 15500, 14500, 13500, 12500, 11500, 10500]
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Feature variables (Age, Mileage, Horsepower)
X = df[['Age', 'Mileage', 'Horsepower']]

# Target variable (Price)
y = df['Price']

# ----- Multiple Linear Regression with Cross-Validation -----
# Create a Linear Regression model
model = LinearRegression()

# Implement K-Fold Cross-Validation (5 splits)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and get the predicted values for each fold
y_pred = cross_val_predict(model, X, y, cv=kf)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse}")

# ----- Visualization: Actual vs Predicted Prices -----
plt.scatter(y, y_pred, color='blue', label='Predicted Prices')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linewidth=2, label='Perfect Prediction Line')

plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices (Multiple Linear Regression)')
plt.legend()
plt.show()

```

## Output:
```
Mean Squared Error: 34172.36073646378

```
![image](https://github.com/user-attachments/assets/6e81c59b-8f69-4cb5-be2f-4680e63d3d1e)


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
