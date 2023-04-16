import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from Digits_functions import *

# Print np array nicely
np.set_printoptions(precision=3, suppress=True)

# Parameters
N_train = 60000     # Number of training samples                 
N_test  = 10000     # Number of test samples
C = 10              # Number of classes
N_pixels = 784      # Number of pixels in image

# Load MNIST data
(train_data, train_label), (test_data, test_label) = mnist.load_data()

# Normalize data so grayscale values are between 0 and 1
train_data = train_data / 255
test_data = test_data / 255

# Calculate mean value of training data for each label
mean_data = mean_ref(train_data, train_label, C, N_pixels)

# Classify test data with nearest neighbor classifier
classified_labels = []
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
    if label == train_data[i]:
        
    classified_labels.append(label)

# Find confusion matrix
confusion_matrix = confusion_matrix_func(classified_labels, test_label, C)
print(confusion_matrix)


# Global to show images
plt.show()