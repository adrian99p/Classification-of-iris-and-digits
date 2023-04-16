import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from Digits_functions import *

# Load MATLAB file
mat_file = scipy.io.loadmat('MNist_ttt4275/data_all.mat')

# Access variables in the MATLAB file
num_test = np.array(mat_file['num_test'])
num_train = np.array(mat_file['num_train'])
testlab = np.array(mat_file['testlab'])
trainlab = np.array(mat_file['trainlab'])
trainv = np.array(mat_file['trainv'])
vec_size = np.array(mat_file['vec_size'])
testv = np.array(mat_file['testv'])

# Read binary data from file
with open('MNist_ttt4275/test_images.bin', 'rb') as f:
    # Read the binary data into a NumPy array
    data = np.fromfile(f, dtype=np.uint8)

pixel_size = 28
# Calculate number of images based on actual size of data
num_images = len(data) // (pixel_size * pixel_size)
if len(data) % (pixel_size * pixel_size) != 0:
    print("Warning: Data size does not match assumed image size.")
    print("Actual number of images may be different.")

# Reshape data into 28x28 matrices with row-major order
images = data[:num_images * pixel_size * pixel_size].reshape(num_images, pixel_size, pixel_size, order='C')

print(images.shape)

# Display 2 images using a for loop
for i in range(2):
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Image {i+1}")
    plt.axis('off')
    plt.show()
