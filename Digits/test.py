import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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
# kmeans = KMeans(n_clusters=K, random_state=0).fit(training_data.reshape(-1, 1))
# kmeans_centers = kmeans.cluster_centers_

# Store cluster labels for training data and cluster centers in a file
# np.savetxt("cluster_labels.txt", kmeans.labels_, fmt="%d")
# np.savetxt("cluster_centers.txt", kmeans_centers, fmt="%f")

# Load cluster labels and cluster centers from file
kmeans_labels = np.loadtxt("cluster_labels.txt", dtype=int)
kmeans_centers = np.loadtxt("cluster_centers.txt", dtype=float)

# Predict cluster labels for test data
# Make test_labels array with integers
test_labels = np.zeros_like(test_data, dtype=int)

for i in range(len(test_data)):
    distances = []
    for j in range(len(kmeans_centers)):
        distance = euclidean_distance(test_data[i], kmeans_centers[j], 1)
        distances.append(distance)
    
    # Find label with smallest distance
    label = np.argmin(distances)
    test_labels[i] = label

# Map cluster labels to class labels
class_labels = np.zeros_like(test_labels)
classes = np.array([0, 1, 2])
for i in range(K):
    cluster_indices = np.where(test_labels == i)
    class_labels[cluster_indices] = int(classes[i % len(classes)])

print("Test Data Labels (Mapped to Classes):")
print(class_labels)
