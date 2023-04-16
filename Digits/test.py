import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Digits_functions import *

# Make training and test data from three different gaussian distributions with different means in one matrix
gauss_1 = np.random.normal(0, 1, 100)
gauss_2 = np.random.normal(50, 1, 100)
gauss_3 = np.random.normal(100, 1, 100)
training_data = np.concatenate((gauss_1, gauss_2, gauss_3))

gauss_11 = np.random.normal(0, 1, 5)
gauss_22 = np.random.normal(50, 1, 5)
gauss_33 = np.random.normal(100, 1, 5)
test_data = np.concatenate((gauss_11, gauss_22, gauss_33))

# Perform k-means clustering on the data
K = 3
kmeans = KMeans(n_clusters=K, random_state=0).fit(training_data.reshape(-1, 1))
kmeans_centers = kmeans.cluster_centers_

# Predict cluster labels for test data
test_labels = kmeans.predict(test_data.reshape(-1, 1))

# Map cluster labels to class labels
class_labels = np.zeros_like(test_labels)
classes = np.array([0, 1, 2])
for i in range(K):
    cluster_indices = np.where(test_labels == i)
    class_labels[cluster_indices] = classes[i % len(classes)]

print("Test Data Labels (Mapped to Classes):")
print(class_labels)
