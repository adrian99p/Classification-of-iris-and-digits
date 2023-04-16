import matplotlib.pyplot as plt
import numpy as np


# Print function
def plot_digit(data_set, index):
    plt.imshow(data_set[index], cmap=plt.get_cmap('gray'))

# Calculate mean value of training data for each label
def mean_ref(train_data, train_label, C, N_pixels):
    mean_data = np.zeros((C, N_pixels))
    for i in range(C):
        mean_data[i] = np.mean(train_data[train_label == i], axis=0).reshape(N_pixels)
    return mean_data

# One hot encoding
def one_hot_encode(label, C):
    one_hot_label = np.zeros((label.shape[0], C))
    for i in range(label.shape[0]):
        one_hot_label[i, label[i]] = 1
    return one_hot_label

# Calculate euclidean distance
def euclidean_distance(x, mean, N_pixels):
    mean = mean.reshape(N_pixels, 1)
    x = x.reshape(N_pixels, 1)
    return ((x - mean).T).dot(x - mean)

def confusion_matrix_func(classified_labels, test_label, C):
    confusion_matrix = np.zeros((C, C))
    for i in range(len(test_label)):
        confusion_matrix[test_label[i], classified_labels[i]] += 1
    return confusion_matrix


# Calculate mahalanobis distance
def mahalanobis_distance(x, mean, cov, N_pixels):
    mean = mean.reshape(N_pixels, 1)
    x = x.reshape(N_pixels, 1)
    return ((x - mean).T).dot(np.linalg.inv(cov)).dot(x - mean)