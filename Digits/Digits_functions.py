import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Calculate mean value of training data for each label
def mean_digit_value_image(train_data, train_label, C, N_pixels):
    mean_data = np.zeros((C, N_pixels))
    for i in range(C):
        mean_data[i] = np.mean(train_data[train_label == i], axis=0).reshape(N_pixels)
    return mean_data

# Calculate euclidean distance
def euclidean_distance(x, mean, N_pixels):
    mean = mean.reshape(N_pixels, 1)
    x = x.reshape(N_pixels, 1)
    return ((x - mean).T).dot(x - mean)

def confusion_matrix_func(classified_labels, test_label, C):
    confusion_matrix = np.zeros((C, C))
    for i in range(len(classified_labels)):
        confusion_matrix[test_label[i], classified_labels[i]] += 1
    return confusion_matrix

# Calculate mahalanobis distance
def mahalanobis_distance(x, mean, cov, N_pixels):
    mean = mean.reshape(N_pixels, 1)
    x = x.reshape(N_pixels, 1)
    return ((x - mean).T).dot(np.linalg.inv(cov)).dot(x - mean)

# Find error rate
def error_rate_func(confusion_matrix):
    error = np.trace(confusion_matrix)
    return round(1 - (error / np.sum(confusion_matrix)),2)

# Plot functions
def plot_digit(data_set, index):
    plt.imshow(data_set[index], cmap=plt.get_cmap('gray'))

def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize = (10,7))
    plt.title('Confusion matrix')
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f') 
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

def plot_classified_image(test_image, mean_image):
    plt.subplot(1, 2, 1)
    plt.imshow(test_image, cmap=plt.get_cmap('gray'))
    plt.title('Test image')
    plt.subplot(1, 2, 2)
    plt.imshow(mean_image, cmap=plt.get_cmap('gray'))
    plt.title('Mean image')
    plt.show()

def compare_test_images(N_plots, test_data, mean_data, classified_labels, labels_indexes):
    plt.figure()
    for i in range(N_plots):

        lab_index = labels_indexes[i]

        test_image = test_data[lab_index]
        predicted_image = mean_data[classified_labels[lab_index]].reshape(28, 28)
        difference_image = test_image - predicted_image

        plt.subplot(N_plots,3,3*i+1)
        plt.imshow(test_image, cmap=plt.get_cmap('gray'))
        if i == 0:
            plt.title('Test image')

        plt.subplot(N_plots,3,3*i+2)
        plt.imshow(predicted_image, cmap=plt.get_cmap('gray'))
        if i == 0:
            plt.title('Predicted image')

        plt.subplot(N_plots,3,3*i+3)
        plt.imshow(difference_image, cmap=plt.get_cmap('gray'))
        if i == 0:
            plt.title('Difference image')

# Plot cluster_to_digit image in sorted order
def plot_cluster_to_digit(cluster_to_digit, kmeans_centers,M_clusters):
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    fig.suptitle(str(M_clusters) + " clusters mapped to a digit ", fontsize=16, fontweight="bold")

    cluster_to_digit_sorted = sorted(cluster_to_digit.items(), key=lambda x: x[1])
    for i in range(len(cluster_to_digit_sorted)):
        digit = cluster_to_digit_sorted[i][1]
        mean_image = kmeans_centers[cluster_to_digit_sorted[i][0]]
        # Subplot mean_image in sorted order depending on M_clusters
        plt.subplot(8, 8, i + 1)
        plt.imshow(mean_image.reshape(28, 28), cmap="gray")
        plt.title(digit, fontsize=8, color="red", fontweight="bold", y=-0.33, x=0.5)
        plt.axis("off")
        
    plt.show()

