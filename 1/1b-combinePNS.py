import pandas as pd
import numpy as np
from scipy import stats

# Create a sample DataFrame
data = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.rand(100),
    'C': np.random.randint(0, 10, 100)
})

# Compute basic statistics using pandas
mean_A = data['A'].mean()
std_dev_B = data['B'].std()

# Perform linear regression using SciPy
slope, intercept, r_value, p_value, std_err = stats.linregress(data['A'], data['B'])

# Apply a function to a column using NumPy
data['C_squared'] = np.square(data['C'])

# Group by and aggregate using pandas
grouped_mean = data.groupby('C')['A'].mean()

# Combine results
print("Mean of column A:", mean_A)
print("Standard deviation of column B:", std_dev_B)
print("Linear Regression Slope:", slope)
print("Linear Regression Intercept:", intercept)
print("R-squared value:", r_value**2)
print("P-value:", p_value)
print("Standard Error:", std_err)
print("Grouped mean by column C:", grouped_mean)
