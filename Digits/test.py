import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from Digits_functions import *




A = np.array([100, 20, 30, 1, 2, 3])
B = np.argsort(A)[:3]

print(B)