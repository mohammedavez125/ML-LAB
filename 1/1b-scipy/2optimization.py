import numpy as np
from scipy import optimize

# Define a quadratic function
def f(x):
    return x**2 + 5*np.sin(x)

# Find the minimum
result = optimize.minimize(f, x0=0)  # Initial guess is x0=0
print("Minimum point:", result.x)
print("Minimum value:", result.fun)
