import scipy.io
import numpy as np

# Load MATLAB file
mat_file = scipy.io.loadmat('MNist_ttt4275/data_all.mat')

# Access variables in the MATLAB file
num_test = np.array(mat_file['num_test'])
num_train = np.array(mat_file['num_train'])
testlab = np.array(mat_file['testlab'])
trainlab = np.array(mat_file['trainlab'])
trainv = np.array(mat_file['trainv'])
vec_size = np.array(mat_file['vec_size'])

# Use the loaded data in Python
print("num_test:")
print(num_test)
print("num_train:")
print(num_train)
print("testlab:")
print(testlab)
print("trainlab:")
print(trainlab)
print("trainv:")
print(trainv)
print("vec_size:")
print(vec_size)