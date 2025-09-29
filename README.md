# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 23-09-2025



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
```
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the data from your CSV file
data = pd.read_csv("/content/heart_rate.csv")

# Select the time series data (in this case, column 'T1')
# You can change 'T1' to 'T2', 'T3', or 'T4' if you want to analyze a different column
time_series = data['T1']

# Perform the Augmented Dickey-Fuller test
result = adfuller(time_series.dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

# Split the data into training and testing sets
train_size = int(len(time_series) * 0.8)
train, test = time_series.iloc[:train_size], time_series.iloc[train_size:]

# Plot ACF and PACF
plot_acf(time_series.dropna(), lags=30)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(time_series.dropna(), lags=30)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Fit the AutoRegressive model
# The number of lags is set to 5, but you may need to adjust this based on the ACF and PACF plots
model = AutoReg(train, lags=5).fit()
print(model.summary())

# Make predictions
preds = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Calculate and print the Mean Squared Error
error = mean_squared_error(test, preds)
print("Mean Squared Error:", error)

# Plot the actual vs. predicted values
plt.figure(figsize=(10,5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, preds, label='Predicted', color='red')
plt.legend()
plt.title("Heart Rate Time Series: Actual vs Predicted")
plt.show()
```
### OUTPUT:

<img width="552" height="402" alt="image" src="https://github.com/user-attachments/assets/0762897c-8edc-460b-9f82-ea07d9e9ca90" />

<img width="560" height="407" alt="image" src="https://github.com/user-attachments/assets/07d10665-f24c-49aa-87d7-c8996ee4f3a4" />

<img width="594" height="480" alt="image" src="https://github.com/user-attachments/assets/177b52fd-6b72-4b8a-a487-088d6649d90b" />

FINIAL PREDICTION

<img width="799" height="423" alt="image" src="https://github.com/user-attachments/assets/e8ac8029-dc9a-430b-b74f-ba536dd12813" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.
