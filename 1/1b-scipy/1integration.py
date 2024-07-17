import numpy as np
from scipy import integrate

# Define a function to integrate
def f(x):
    return np.sin(x)

# Integrate the function from 0 to pi
result, error = integrate.quad(f, 0, np.pi)
print("Integral result:", result)
print("Estimated error:", error)
