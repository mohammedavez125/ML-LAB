import numpy as np
from scipy.linalg import solve, eig

# Solving a System of Linear Equations
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

# Solve the system of equations
x = solve(A, b)
print("Solution of the system:", x)

# Eigenvalues and Eigenvectors
# Define a matrix
A = np.array([[3, 1], [1, 2]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
