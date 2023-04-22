import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from Digits_functions import *
from sklearn.cluster import KMeans
import time

# Print np array nicely
np.set_printoptions(precision=3, suppress=True)

# Parameters
N_train = 60000                   # Number of training samples                 
N_test  = 10000                    # Number of test samples
C = 10                            # Number of classes
K_neighbors = 7                   # Number of nearest neighbors
M_clusters  = 64                  # Number of clusters
N_pixels    = 784                 # Number of pixels in image

# Classification methods
NN_classification      = False     # Use the nearest neighbor classifier
Kmeans_classification  = False    # Use NN with k-means clustering classifier
KNN_classification     = True    # Use k-nearest neighbor classifier with k-means clustering


# Plot parameters
visualize_confusion_matrix = True  # Visualize confusion images
visualize_NN_comparison    = True  # Visualize nearest neighbor comparison test, prediction
N_Comparisons = 5                  # Number of comparisons to visualize

# Load MNIST hand written digit data
(train_data, train_label), (test_data, test_label) = mnist.load_data()

# Select only N_train and N_test data and labels
train_data  = train_data[:N_train]
train_label = train_label[:N_train]
test_data   = test_data[:N_test]
test_label  = test_label[:N_test]

# Normalize data so grayscale values are between 0 and 1
train_data = train_data / 255
test_data  = test_data / 255

# Classify test data with nearest neighbor classifier -------------------------------------------------------------------
if NN_classification:
    print("NN classification")
    
    print("Start training")
    time_start = time.time()

    classified_labels = []
    correct_labels_indexes = []
    failed_labels_indexes = []
  
    # Calculate distances for each test data to each training data
    for i in range(N_test):
        # Get test image
        test_image = test_data[i]

        distances = []
        for j in range(N_train):
            train_image = train_data[j]
            distance = euclidean_distance(test_image, train_image, N_pixels)
            distances.append(distance)
        
        # Find label with smallest distance 
        closest_test_data_index = np.argmin(distances)
        label = train_label[closest_test_data_index]       

        if label == test_label[i]:
            correct_labels_indexes.append(i)
        else:
            failed_labels_indexes.append(i)
        classified_labels.append(label)

    # Print training time
    time_end = time.time()
    training_time = int(time_end - time_start)
    print_time(time_start, time_end)

    # Find confusion matrix
    confusion_matrix = confusion_matrix_func(classified_labels, test_label, C)
    print("Confusion matrix: ")
    print(confusion_matrix)

    # Print error rate
    error_rate = error_rate_func(confusion_matrix)
    print("Error rate: ", error_rate*100, "%")

    # Save confusion matrix to file
    save_to_file("NN/CM_NN_", confusion_matrix, error_rate, N_train, N_test, training_time)

    # Visualize confusion matrix
    plot_confusion_matrix("NN", confusion_matrix, error_rate, visualize_confusion_matrix)

    # Visualize nearest neighbor comparison test, prediction
    plot_NN_comparison(test_data, test_label, classified_labels, correct_labels_indexes, failed_labels_indexes, N_Comparisons, visualize_NN_comparison)

# Classify test data with Kmeans classifier----------------------------------------------------------------------------------------------------------------------
if Kmeans_classification:
    print("K-means classification")
    
    print("Start training")
    time_start = time.time()
    
    # Perform k-means clustering on training data 
    start_training = False
    if start_training:

        # Create 64 clusters for each unique label from training data
        kmeans_centers = np.empty((0, N_pixels))
        cluster_labels = np.empty((0, 1))

        for i in range(C):
            # Get indices for label i
            label_indices = np.where(train_label == i)[0]
            # Get data for label i
            label_data = train_data[label_indices]
            # Perform k-means clustering on label data
            kmeans = KMeans(n_clusters=M_clusters, random_state=0).fit(label_data.reshape(len(label_indices), N_pixels))
            # Store cluster centers
            kmeans_centers = np.append(kmeans_centers, kmeans.cluster_centers_, axis=0)
            # Append M_clusters cluster labels to cluster_labels
            cluster_labels = np.append(cluster_labels, np.full((M_clusters, 1), i), axis=0)

        #Store cluster labels for training data and cluster centers in a file in a folder called "kmeans_trained"
        np.savetxt("kmeans_trained/cluster_labels.txt", cluster_labels, fmt="%d")
        np.savetxt("kmeans_trained/cluster_centers.txt", kmeans_centers, fmt="%f")

    # Load cluster labels and cluster centers from file
    cluster_labels = np.loadtxt("kmeans_trained/cluster_labels.txt", dtype=int)
    kmeans_centers = np.loadtxt("kmeans_trained/cluster_centers.txt", dtype=float)

    # Classify test data with nearest neighbor classifier
    classified_labels = []
   
    # Calculate distances for each test data
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
        label = cluster_labels[label]
        classified_labels.append(label)
    
    # Print training time
    time_end = time.time()
    training_time = int(time_end - time_start)
    print_time(time_start, time_end)

    # Find confusion matrix
    confusion_matrix = confusion_matrix_func(classified_labels, test_label, C)
    print(confusion_matrix)

    # Print error rate
    error_rate = error_rate_func(confusion_matrix)
    print("Error rate: ", error_rate*100, "%")

    # Save confusion matrix to file
    save_to_file("Kmeans/CM_Kmeans_", confusion_matrix ,error_rate, N_train, N_test, training_time)

    # Plot confusion matrix
    plot_confusion_matrix("Kmeans", confusion_matrix, error_rate, visualize_confusion_matrix)

# Classify test data with KNN classifier----------------------------------------------------------------------------------------------------------------------
if KNN_classification:
    print("KNN classification")

    print("Start training")
    time_start = time.time()

    # Perform k-means clustering on training data 
    start_training = False
    if start_training:

        # Create 64 clusters for each unique label from training data
        kmeans_centers = np.empty((0, N_pixels))
        cluster_labels = np.empty((0, 1))

        for i in range(C):
            # Get indices for label i
            label_indices = np.where(train_label == i)[0]
            # Get data for label i
            label_data = train_data[label_indices]
            # Perform k-means clustering on label data
            kmeans = KMeans(n_clusters=M_clusters, random_state=0).fit(label_data.reshape(len(label_indices), N_pixels))
            # Store cluster centers
            kmeans_centers = np.append(kmeans_centers, kmeans.cluster_centers_, axis=0)
            # Append M_clusters cluster labels to cluster_labels
            cluster_labels = np.append(cluster_labels, np.full((M_clusters, 1), i), axis=0)

        #Store cluster labels for training data and cluster centers in a file in a folder called "kmeans_trained"
        np.savetxt("kmeans_trained/cluster_labels.txt", cluster_labels, fmt="%d")
        np.savetxt("kmeans_trained/cluster_centers.txt", kmeans_centers, fmt="%f")

    # Load cluster labels and cluster centers from file
    cluster_labels = np.loadtxt("kmeans_trained/cluster_labels.txt", dtype=int)
    kmeans_centers = np.loadtxt("kmeans_trained/cluster_centers.txt", dtype=float)

    # Classify test data using K-nearest neighbor classifier
    classified_labels = []

    for i in range(N_test):
        # Get test image
        test_image = test_data[i]

        distances = np.zeros(len(kmeans_centers))
        for j in range(len(kmeans_centers)):
            mean_image = kmeans_centers[j]
            distance = euclidean_distance(test_image, mean_image, N_pixels)
            distances[j] = distance

        nearest_neighbors = np.argsort(distances)[:K_neighbors]

        nearest_neighbors_labels = []
        for neighbor in nearest_neighbors:
            nearest_neighbors_labels.append(cluster_labels[neighbor])

        # Find label with most occurences
        label = np.argmax(np.bincount(nearest_neighbors_labels))
        classified_labels.append(label)

    # Print training time
    time_end = time.time()
    training_time = int(time_end - time_start)
    print_time(time_start, time_end)

    # Find confusion matrix
    confusion_matrix = confusion_matrix_func(classified_labels, test_label, C)
    print(confusion_matrix)

    error_rate = error_rate_func(confusion_matrix)
    print("Error rate: ", error_rate*100, "%")

    # Save confusion matrix to file
    save_to_file("KNN/CM_KNN_", confusion_matrix ,error_rate, N_train, N_test, training_time)

    # Plot confusion matrix
    plot_confusion_matrix("KNN, K=" + str(K_neighbors),confusion_matrix, error_rate, visualize_confusion_matrix)

# ---------------------------------------------------------------------------------------------------------------------
plt.show()