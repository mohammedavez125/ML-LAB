import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Load the dataset
df_sal = pd.read_csv('C:/Users/moham/Downloads/ml_lab/2/content/Salary_Data.csv')
print(df_sal.head())

# Step 3: Data analysis
print(df_sal.describe())

# Data distribution
plt.title('Salary Distribution Plot')
sns.histplot(df_sal['Salary'], kde=True)
plt.show()

# Relationship between Salary and Experience
plt.scatter(df_sal['YearsExperience'], df_sal['Salary'], color='lightcoral')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.box(False)
plt.show()

# Step 4a: Split the dataset into dependent/independent variables
X = df_sal.iloc[:, :1]  # independent
y = df_sal.iloc[:, 1:]  # dependent

# Step 4b: Split data into Train/Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 5: Train the regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 6: Predict the result
y_pred_test = regressor.predict(X_test)  # predicted value of y_test
y_pred_train = regressor.predict(X_train)  # predicted value of y_train

# Step 7: Plot the training and test results
# Prediction on training set
plt.scatter(X_train, y_train, color='lightcoral')
plt.plot(X_train, y_pred_train, color='firebrick')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['Prediction', 'Actual'], title='Legend', loc='best', facecolor='white')
plt.box(False)
plt.show()

# Prediction on test set
plt.scatter(X_test, y_test, color='lightcoral')
plt.plot(X_train, y_pred_train, color='firebrick')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['Prediction', 'Actual'], title='Legend', loc='best', facecolor='white')
plt.box(False)
plt.show()

# Regressor coefficients and intercept
print(f'Coefficient: {regressor.coef_}')
print(f'Intercept: {regressor.intercept_}')
