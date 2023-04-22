import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from Digits_functions import *
from sklearn.cluster import KMeans
import time

train_data = [60,53,5,66,44,67,90,8,9,10]

train_label = [0, 0, 2, 4, 4, 0, 6, 2, 2, 9]


# Sort arrays based on train_label
sorted_indices = np.argsort(train_label)
train_label_sorted = train_label[sorted_indices]
train_data_sorted = train_data[sorted_indices]


print(train_data_sorted)
print(train_label_sorted)