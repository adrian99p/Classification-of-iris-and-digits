import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from Digits_functions import *
from sklearn.cluster import KMeans
import time

# Print np array nicely
np.set_printoptions(precision=3, suppress=True)

# Parameters
N_train = 60000                      # Number of training samples                 
N_test  = 10000                     # Number of test samples
C = 10                               # Number of classes
K = 7                                 # Number of nearest neighbors
N_pixels = 784                       # Number of pixels in image
M_clusters = 640                      # Number of clusters
visualize_confusion_matrix = False   # Visualize confusion images
N_Comparisons = 5                    # Number of comparisons to visualize
visualize_NN_comparison = False      # Visualize nearest neighbor comparison test, prediction

NN_mean_classification = False             # Use nearest neighbor classifier
NN_actual_classification = False           # Use the actual nearest neighbor classifier
Kmeans_classification =  True             # Use k-means clustering classifier
KNN_classification = False                 # Use k-nearest neighbor classifier

# Load MNIST hand written digit data
(train_data, train_label), (test_data, test_label) = mnist.load_data()

# Normalize data so grayscale values are between 0 and 1
train_data = train_data / 255
test_data = test_data / 255

# Classify test data with nearest neighbor classifier -------------------------------------------------------------------
if NN_mean_classification:
    print("NN mean classification")
    # Calculate mean value of training data for each label
    mean_data = mean_digit_value_image(train_data, train_label, C, N_pixels)
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
    print("Error rate: ", error_rate*100, "%")    

# Classify test data with actual nearestest neighbor classifier -------------------------------------------------------------------
if NN_actual_classification:
    print("Actual NN classification")
    classified_labels = []
    correct_labels_indexes = []
    failed_labels_indexes = []

    print("Start training")
    time_start = time.time()
    # Calculate distance matrix
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

    time_end = time.time()
    # Convert to hours, minutes and seconds
    time_end = time.time()
    time_elapsed = time_end - time_start
    hours = int(time_elapsed // 3600)
    minutes = int((time_elapsed % 3600) // 60)
    seconds = int(time_elapsed % 60)
    print("Training time: ", hours, "h", minutes, "m", seconds, "s")

    # Find confusion matrix
    confusion_matrix = confusion_matrix_func(classified_labels, test_label, C)
    print("Confusion matrix: ")
    print(confusion_matrix)

    # Print error rate
    error_rate = error_rate_func(confusion_matrix)
    print("Error rate: ", error_rate*100, "%") 
    # Save confusion matrix to file
    np.savetxt("confusion_matrix_actual_NN.txt", confusion_matrix, fmt="%d")

    # Visualize confusion matrix
    plot_confusion_matrix(confusion_matrix, error_rate)

# ----------------------------------------------------------------------------------------------------------------------
if Kmeans_classification:
    print("K-means classification")
    # Perform k-means clustering on training data 
    start_training = True
    if start_training:
        kmeans = KMeans(n_clusters=M_clusters, random_state=0).fit(train_data.reshape(N_train, N_pixels))
        kmeans_centers = kmeans.cluster_centers_

        #Store cluster labels for training data and cluster centers in a file in a folder called "Digits"
        #np.savetxt("kmeans_trained/cluster_labels.txt", kmeans.labels_, fmt="%d")
        #np.savetxt("kmeans_trained/cluster_centers.txt", kmeans_centers, fmt="%f")

    # Load cluster labels and cluster centers from file
    #cluster_labels = np.loadtxt("kmeans_trained/cluster_labels.txt", dtype=int)
    #kmeans_centers = np.loadtxt("kmeans_trained/cluster_centers.txt", dtype=float)
    cluster_labels = kmeans.labels_
    kmeans_centers = kmeans.cluster_centers_

    # Map cluster labels to digit labels using majority voting method
    digit_labels = train_label
    cluster_to_digit = {}
    for cluster_label in range(len(kmeans_centers)):
        cluster_digit_labels = digit_labels[cluster_labels == cluster_label]
        majority_digit_label = np.argmax(np.bincount(cluster_digit_labels))
        cluster_to_digit[cluster_label] = majority_digit_label

    # Plot cluster_to_digit image in sorted order
    #plot_cluster_to_digit(cluster_to_digit, kmeans_centers,M_clusters)

    # Classify test data with nearest neighbor classifier
    classified_labels = []
    correct_labels_indexes = []
    failed_labels_indexes = []
    actual_labels = []
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
        classified_labels.append(label)
        actual_labels.append(test_label[i])

        if label == test_label[i]:
            correct_labels_indexes.append(i)
        else:
            failed_labels_indexes.append(i)
        

    # Find confusion matrix
    confusion_matrix = confusion_matrix_func(classified_labels, test_label, C)

    # Print confusion matrix
    print(confusion_matrix)

    # Print error rate
    error_rate = error_rate_func(confusion_matrix)
    print("Error rate: ", error_rate*100, "%")

# ----------------------------------------------------------------------------------------------------------------------
if KNN_classification:
    print("KNN classification")
    # Perform k-means clustering on training data 
    start_training = False
    if start_training:
        kmeans = KMeans(n_clusters=M_clusters, random_state=0).fit(train_data.reshape(N_train, N_pixels))
        kmeans_centers = kmeans.cluster_centers_

        #Store cluster labels for training data and cluster centers in a file in a folder called "Digits"
        np.savetxt("kmeans_trained/cluster_labels.txt", kmeans.labels_, fmt="%d")
        np.savetxt("kmeans_trained/cluster_centers.txt", kmeans_centers, fmt="%f")

    # Load cluster labels and cluster centers from file
    cluster_labels = np.loadtxt("kmeans_trained/cluster_labels.txt", dtype=int)
    kmeans_centers = np.loadtxt("kmeans_trained/cluster_centers.txt", dtype=float)

    # Map cluster labels to digit labels using majority voting method
    digit_labels = train_label
    cluster_to_digit = {}
    for cluster_label in range(len(kmeans_centers)):
        cluster_digit_labels = digit_labels[cluster_labels == cluster_label]
        majority_digit_label = np.argmax(np.bincount(cluster_digit_labels))
        cluster_to_digit[cluster_label] = majority_digit_label

    # Plot cluster_to_digit image in sorted order
    #plot_cluster_to_digit(cluster_to_digit, kmeans_centers,M_clusters)

    # Classify test data using K-nearest neighbor classifier
    classified_labels = []
    correct_labels_indexes = []
    failed_labels_indexes = []
    actual_labels = []
    for i in range(N_test):
        # Get test image
        test_image = test_data[i]

        distances = np.zeros(len(kmeans_centers))
        for j in range(len(kmeans_centers)):
            mean_image = kmeans_centers[j]
            distance = euclidean_distance(test_image, mean_image, N_pixels)
            distances[j] = distance

        nearest_neighbors = np.argsort(distances)[:K]

        nearest_neighbors_labels = []
        for neighbor in nearest_neighbors:
            nearest_neighbors_labels.append(cluster_to_digit[neighbor])

        # Find label with most occurences
        label = np.argmax(np.bincount(nearest_neighbors_labels))
        classified_labels.append(label)
        actual_labels.append(test_label[i])

        if label == test_label[i]:
            correct_labels_indexes.append(i)
        else:
            failed_labels_indexes.append(i)

    # Find confusion matrix
    confusion_matrix = confusion_matrix_func(classified_labels, test_label, C)
    print(confusion_matrix)

    error_rate = error_rate_func(confusion_matrix)
    print("Error rate: ", error_rate*100, "%")


# ----------------------------------------------------------------------------------------------------------------------































plt.show()