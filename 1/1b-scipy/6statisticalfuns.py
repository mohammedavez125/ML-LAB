import numpy as np
from scipy import stats

# Sample data
data = np.array([2, 8, 3, 5, 7, 4, 6, 8, 5, 9])

# Compute mean, median, and standard deviation
mean = np.mean(data)
median = np.median(data)
std_dev = np.std(data)
mode = stats.mode(data)
print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)
print("Mode:", mode)
