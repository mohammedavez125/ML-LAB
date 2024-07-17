import numpy as np

# Create a 1D array
array_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:", array_1d)

# Create a 2D array
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", array_2d)

# Array Attributes
print("Shape of 1D array:", array_1d.shape)
print("Shape of 2D array:", array_2d.shape)
print("Data type of 1D array:", array_1d.dtype)

# 2. Array Operations
# Element-wise Operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise addition
print("Addition:", a + b)

# Element-wise multiplication
print("Multiplication:", a * b)

# Element-wise square
print("Square:", a ** 2)

# Aggregate Functions
array = np.array([1, 2, 3, 4, 5])
print("Sum:", np.sum(array))
print("Mean:", np.mean(array))
print("Standard Deviation:", np.std(array))
print("Max:", np.max(array))
print("Min:", np.min(array))

# 3. Indexing and Slicing
# Indexing
array = np.array([1, 2, 3, 4, 5])
print("First element:", array[0])
print("Last element:", array[-1])

# 2D array indexing
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("Element at (0,1):", array_2d[0, 1])

# Slicing
# 1D array slicing
print("Slice [1:4]:", array[1:4])

# 2D array slicing
print("First row:", array_2d[0, :])
print("Second column:", array_2d[:, 1])

# 4. Linear Algebra Operations
# Matrix Multiplication
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Matrix multiplication
result = np.dot(a, b)
print("Matrix Multiplication:\n", result)

# Determinant and Inverse
matrix = np.array([[1, 2], [3, 4]])

# Determinant
det = np.linalg.det(matrix)
print("Determinant:", det)

# Inverse
inv = np.linalg.inv(matrix)
print("Inverse:\n", inv)

# 5. Random Number Generation
# Generating Random Numbers
# Random numbers between 0 and 1
random_array = np.random.rand(5)
print("Random numbers:", random_array)

# Random integers
random_integers = np.random.randint(0, 10, size=5)
print("Random integers:", random_integers)

# Random normal distribution
random_normal = np.random.randn(5)
print("Random normal distribution:", random_normal)

# 6. Broadcasting
# Broadcasting Example
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])

# Broadcasting addition
result = a + b
print("Broadcasting result:\n", result)

# 7. Reshaping and Transposing
# Reshaping Arrays
array = np.array([[1, 2, 3], [4, 5, 6]])
reshaped_array = array.reshape((3, 2))
print("Original array:\n", array)
print("Reshaped array:\n", reshaped_array)

# Transposing Arrays
transposed_array = array.T
print("Transposed array:\n", transposed_array)

# 8. Filtering and Conditional Selection
# Conditional Selection
array = np.array([1, 2, 3, 4, 5, 6])

# Selecting elements greater than 3
condition = array > 3
filtered_array = array[condition]
print("Elements greater than 3:", filtered_array)
