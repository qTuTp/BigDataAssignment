import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Database import getDatabase
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Connect to the database and retrieve the collections
db = getDatabase()
collection = db["caseVaxMalaysia"]
# collCaseM = db["caseMalaysiaClean"]
# collVaxM = db["vaxMalaysiaClean"]

# # Retrieve data from MongoDB collections and load into pandas DataFrames
# dataCaseM = pd.DataFrame(list(collCaseM.find()))
# dataVaxM = pd.DataFrame(list(collVaxM.find()))
data = pd.DataFrame(list(collection.find()))


# # Convert dates to datetime format
# dataCaseM['date'] = pd.to_datetime(dataCaseM['date'])
# dataVaxM['date'] = pd.to_datetime(dataVaxM['date'])

# # Merge the cases and vaccination data on the date
# data = pd.merge(dataCaseM, dataVaxM, on='date')

# Calculate cumulative vaccinations
data['cumulative_vaccinations'] = data['daily'].cumsum()

# X: Cumulative Vaccinations, Y: Daily New Cases
X = data[['cumulative_vaccinations']].values
Y = data[['cases_new']].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

Y_train_pred = model.predict(X_train)

# Plot the training data and the regression line
plt.figure(figsize=(12, 8))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, Y_train_pred, color='red', label='Regression Line')
plt.title('Linear Regression - Training Set')
plt.xlabel('Cumulative Vaccinations (in millions)')
plt.ylabel('Daily New Cases')
plt.legend()
# Customize the x-axis tick labels to show values in millions
ax = plt.gca()
ax.set_xticklabels(['{:.1f}M'.format(x / 1e6) for x in ax.get_xticks()])
plt.show()

# Plot the test data and the regression line
plt.figure(figsize=(12, 8))
plt.scatter(X_test, y_test, color='blue', label='Test Data')
plt.plot(X_train, Y_train_pred, color='red', label='Regression Line')
plt.title('Linear Regression - Test Set')
plt.xlabel('Cumulative Vaccinations (in millions)')
plt.ylabel('Daily New Cases')
plt.legend()
# Customize the x-axis tick labels to show values in millions
ax = plt.gca()
ax.set_xticklabels(['{:.1f}M'.format(x / 1e6) for x in ax.get_xticks()])
plt.show()