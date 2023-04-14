import numpy as np

# Create a NumPy array


A = np.array([[1, 2, 3]])
B = np.array([[5, 5, 5, 5, 5]])
C = A.T @ B


print(C)
