import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from Digits_functions import *
from sklearn.cluster import KMeans


# Print np array nicely
np.set_printoptions(precision=3, suppress=True)

# Parameters
N_train = 60000                      # Number of training samples                 
N_test  = 10000                      # Number of test samples
C = 10                               # Number of classes
N_pixels = 784                       # Number of pixels in image
visualize_confusion_matrix = False   # Visualize mean images
N_Comparisons = 5                    # Number of comparisons to visualize
visualize_NN_comparison = False      # Visualize nearest neighbor comparison

# Load MNIST data
(train_data, train_label), (test_data, test_label) = mnist.load_data()

# Normalize data so grayscale values are between 0 and 1
train_data = train_data / 255
test_data = test_data / 255

# Calculate mean value of training data for each label
mean_data = mean_ref(train_data, train_label, C, N_pixels)

# Classify test data with nearest neighbor classifier
classified_labels = []
correct_labels_indexes = []
failed_labels_indexes = []
for i in range(N_test):
    # Get test image
    test_image = test_data[i]

    distances = []
    for j in range(C):
        mean_image = mean_data[j]
        distance = euclidean_distance(test_image, mean_image, N_pixels)
        distances.append(distance)
    
    # Find label with smallest distance
    label = np.argmin(distances)
    if label == test_label[i]:
        correct_labels_indexes.append(i)
    else:
        failed_labels_indexes.append(i)
    classified_labels.append(label)

# Find confusion matrix
confusion_matrix = confusion_matrix_func(classified_labels, test_label, C)
print(confusion_matrix)

# Print error rate
error_rate = error_rate_func(confusion_matrix)
print("Error rate: ", error_rate)


# Perform k-means clustering on training data
kmeans = KMeans(n_clusters=64, random_state=0).fit(train_data.reshape(N_train, N_pixels))
kmeans_centers = kmeans.cluster_centers_
print(kmeans_centers.shape)

# Classify test data with nearest neighbor classifier
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
classified_labels = []
correct_labels_indexes = []
failed_labels_indexes = []
for i in range(N_test):
    # Get test image
    test_image = test_data[i]

    distances = []
    for j in range(len(kmeans_centers)):
        mean_image = kmeans_centers[j]
        distance = euclidean_distance(test_image, mean_image, N_pixels)
        distances.append(distance)
    
    # Find label with smallest distance
    label = np.argmin(distances)
    
    
# Visualize confusion matrix
if visualize_confusion_matrix:
    plot_confusion_matrix(confusion_matrix)

if visualize_NN_comparison:
    compare_test_images(N_Comparisons, test_data, mean_data, classified_labels, failed_labels_indexes)
    compare_test_images(N_Comparisons, test_data, mean_data, classified_labels, correct_labels_indexes)

plt.show()