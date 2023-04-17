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
visualize_confusion_matrix = False   # Visualize confusion images
N_Comparisons = 5                    # Number of comparisons to visualize
visualize_NN_comparison = False      # Visualize nearest neighbor comparison test, prediction
NN_active = False                    # Use nearest neighbor classifier
Kmeans_active = True                 # Use k-means clustering classifier

# Load MNIST data
(train_data, train_label), (test_data, test_label) = mnist.load_data()

# Normalize data so grayscale values are between 0 and 1
train_data = train_data / 255
test_data = test_data / 255

# Classify test data with nearest neighbor classifier -------------------------------------------------------------------
# Calculate mean value of training data for each label
if NN_active:
    mean_data = mean_ref(train_data, train_label, C, N_pixels)
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

# ----------------------------------------------------------------------------------------------------------------------

# Perform k-means clustering on training data --------------------------------------------------------------------------
# kmeans = KMeans(n_clusters=64, random_state=0).fit(train_data.reshape(N_train, N_pixels))
# kmeans_centers = kmeans.cluster_centers_

# Store cluster labels for training data and cluster centers in a file in a folder called "Digits"
# np.savetxt("kmeans_trained/cluster_labels.txt", kmeans.labels_, fmt="%d")
# np.savetxt("kmeans_trained/cluster_centers.txt", kmeans_centers, fmt="%f")

# Load cluster labels and cluster centers from file
kmeans_labels = np.loadtxt("kmeans_trained/cluster_labels.txt", dtype=int)
kmeans_centers = np.loadtxt("kmeans_trained/cluster_centers.txt", dtype=float)

# Map cluster labels to digit labels using majority voting method
cluster_labels = kmeans_labels
digit_labels = train_label
cluster_to_digit = {}
for cluster_label in range(len(kmeans_centers)):
    cluster_digit_labels = digit_labels[cluster_labels == cluster_label]
    majority_digit_label = np.argmax(np.bincount(cluster_digit_labels))
    cluster_to_digit[cluster_label] = majority_digit_label

# Plot cluster_to_digit image in sorted order
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
fig.suptitle("64 clusters mapped to a digit ", fontsize=16, fontweight="bold")

cluster_to_digit_sorted = sorted(cluster_to_digit.items(), key=lambda x: x[1])
for i in range(len(cluster_to_digit_sorted)):
    digit = cluster_to_digit_sorted[i][1]
    mean_image = kmeans_centers[cluster_to_digit_sorted[i][0]]
    plt.subplot(8, 8, i + 1)
    plt.imshow(mean_image.reshape(28, 28), cmap="gray")
    plt.title(digit, fontsize=8, color="red", fontweight="bold", y=-0.33, x=0.5)
    plt.axis("off")
plt.show()

# Classify test data with nearest neighbor classifier
classified_labels = []
correct_labels_indexes = []
failed_labels_indexes = []
test_labels = []
actual_labels = []
N_test = 20
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
    label = cluster_to_digit[label]
    test_labels.append(label)
    actual_labels.append(test_label[i])

print("K-means clustering")
print("Number of clusters: ", len(kmeans_centers))
print("Labels: ")
print(test_labels)
print("Actual labels: ")
print(actual_labels)

# Find confusion matrix
confusion_matrix = confusion_matrix_func(test_labels, actual_labels, C)
#print(confusion_matrix)

# Print error rate
error_rate = error_rate_func(confusion_matrix)
print("Error rate: ", error_rate*100, "%")

# Plot confusion matrix
#plot_confusion_matrix(confusion_matrix)

# ----------------------------------------------------------------------------------------------------------------------

# Visualize confusion matrix
if visualize_confusion_matrix:
    plot_confusion_matrix(confusion_matrix)

if visualize_NN_comparison:
    compare_test_images(N_Comparisons, test_data, mean_data, classified_labels, failed_labels_indexes)
    compare_test_images(N_Comparisons, test_data, mean_data, classified_labels, correct_labels_indexes)

plt.show()