import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Iris_func import *
import math as m
import time
import seaborn as sns
from scipy.stats import gaussian_kde

# Parameters
D = 4                     # Number of features
C = 3                     # Number of classes
N_train = 30              # Number of training data
N_test = 20               # Number of test data
first_30_to_train = True  # Use first 30 data points for training and last 20 for testing
disabled_features = ['SW', 'SL', 'PL'] # Which features to disable
D = D - len(disabled_features) # Update D
# Dictionary mapping feature keys to integer values
feature_to_int_dict = {
    "SL": 0,
    "SW": 1,
    "PL": 2,
    "PW": 3,
}
disabled_features_int = [feature_to_int_dict[feature] for feature in disabled_features]

# Load seperate iris data
setosa = pd.read_csv('Data/class_1.csv', header=0)
versicolour = pd.read_csv('Data/class_2.csv', header=0)
verginica = pd.read_csv('Data/class_3.csv', header=0)

# Remove unwanted features
setosa.columns = setosa.columns.str.strip()
versicolour.columns = versicolour.columns.str.strip()
verginica.columns = verginica.columns.str.strip()
setosa = setosa.drop(columns=disabled_features)
versicolour = versicolour.drop(columns=disabled_features)
verginica = verginica.drop(columns=disabled_features)

# Create training and test data
if first_30_to_train:
    train_data = pd.concat([setosa[:N_train], versicolour[:N_train], verginica[:N_train]])
    test_data = pd.concat([setosa[N_train:N_train+N_test], versicolour[N_train:N_train+N_test], verginica[N_train:N_train+N_test]])
    train_data = train_data.values
    test_data = test_data.values
else:
    test_data = pd.concat([setosa[:N_test], versicolour[:N_test], verginica[:N_test]])
    train_data = pd.concat([setosa[N_test:N_test+N_train], versicolour[N_test:N_test+N_train], verginica[N_test:N_test+N_train]])
    train_data = train_data.values
    test_data = test_data.values

# Normalizing the data
max_features_val = np.array([train_data[:,i].max() for i in range(D)])
normal_train_data = train_data/max_features_val

# Target vectors
t1 = np.array([1, 0, 0])
t2 = np.array([0, 1, 0])
t3 = np.array([0, 0, 1])
label_train = np.vstack((np.tile(t1, (N_train, 1)), np.tile(t2, (N_train, 1)), np.tile(t3, (N_train, 1))))
label_test = np.vstack((np.tile(t1, (N_test, 1)), np.tile(t2, (N_test, 1)), np.tile(t3, (N_test, 1))))

# Training the LDC
W = np.zeros((C, D+1))
training = True
iterations = 1000
MSE_list = []
print("Starting training")
start_time = time.time()

for i in range(iterations): 
    grad_W_MSE = 0
    MSE = 0
    for i in range(C*N_train):
        # Using 3.2 in compendium 
        x_k  = np.array(train_data[i, :])
        x_k = np.append(x_k, 1)
        z_k = W @ x_k
        g_k = sigmoid(z_k)
        t_k = label_train[i, :]
        grad_W_MSE += grad_W_MSE_func(g_k, t_k, x_k, D)
        MSE += 0.5*(g_k-t_k).T @ (g_k-t_k)
    MSE_list.append(MSE)
    
    alpha = 0.01
    W = W - alpha*grad_W_MSE
    
end_time = time.time()
elapsed_time = end_time - start_time
print("Training time: ", elapsed_time)
print("Training done")

print("W: ", W)

# Find confusion matrix for training data
confusion_matrix_train = np.zeros((C, C))
for i in range(C*N_train):
    x_k  = np.array(train_data[i, :])
    x_k = np.append(x_k, 1)
    z_k = W @ x_k
    g_k = sigmoid(z_k)
    t_k = label_train[i, :]
    if np.argmax(g_k) == np.argmax(t_k):
        confusion_matrix_train[np.argmax(t_k), np.argmax(t_k)] += 1
    else:
        confusion_matrix_train[np.argmax(t_k), np.argmax(g_k)] += 1

print("Confusion matrix for training data using: ")
print(confusion_matrix_train)

# Calculate accuracy in percent
accuracy_train = np.sum(np.diag(confusion_matrix_train))/np.sum(confusion_matrix_train)
print("Accuracy for training data: ", round(accuracy_train,4))
print("Error rate for training data: ", round(1-accuracy_train,4))

# Find confusion matrix for test data
confusion_matrix_test = np.zeros((C, C))
for i in range(C*N_test):
    x_k  = np.array(test_data[i, :])
    x_k = np.append(x_k, 1)
    z_k = W @ x_k
    g_k = sigmoid(z_k)
    t_k = label_test[i, :]
    if np.argmax(g_k) == np.argmax(t_k):
        confusion_matrix_test[np.argmax(t_k), np.argmax(t_k)] += 1
    else:
        confusion_matrix_test[np.argmax(t_k), np.argmax(g_k)] += 1

print("Confusion matrix for test data: ")
print(confusion_matrix_test)

# Calculate accuracy
accuracy_test = np.sum(np.diag(confusion_matrix_test))/np.sum(confusion_matrix_test)
print("Accuracy for test data: ", round(accuracy_test, 4))
print("Error rate for test data: ", round(1-accuracy_test, 4))


# END OF TASK 1
#-----------------------------------------------------------------------------------------
# START OF TASK 2

# Plotting confusion matrices for training and test data
class_labels = ['Setosa', 'Versicolour', 'Veriginica']
df_cm_test = pd.DataFrame(confusion_matrix_test, index = [i for i in class_labels], columns = [i for i in class_labels])
plt.figure(figsize = (10,7))
plt.title("Confusion matrix for test data using first 30")
sns.heatmap(df_cm_test, annot=True)

df_cm_train = pd.DataFrame(confusion_matrix_train, index = [i for i in class_labels], columns = [i for i in class_labels])
plt.figure(figsize = (10,7))
plt.title("Confusion matrix for training data using first 30")
sns.heatmap(df_cm_train, annot=True)



# Plot 3 histograms for feature x for all classes

# Extract features from training data
# feature_1_class_1 = np.array(train_data[:N_train, 0])
# feature_2_class_1 = np.array(train_data[:N_train, 1])
# feature_3_class_1 = np.array(train_data[:N_train, 2])
# feature_4_class_1 = np.array(train_data[:N_train, 3])

# feature_1_class_2 = np.array(train_data[N_train:2*N_train, 0])
# feature_2_class_2 = np.array(train_data[N_train:2*N_train, 1])
# feature_3_class_2 = np.array(train_data[N_train:2*N_train, 2])
# feature_4_class_2 = np.array(train_data[N_train:2*N_train, 3])

# feature_1_class_3 = np.array(train_data[2*N_train:3*N_train, 0])
# feature_2_class_3 = np.array(train_data[2*N_train:3*N_train, 1])
# feature_3_class_3 = np.array(train_data[2*N_train:3*N_train, 2])
# feature_4_class_3 = np.array(train_data[2*N_train:3*N_train, 3])

# feature_plot_1 = [feature_1_class_1, feature_1_class_2, feature_1_class_3]
# feature_plot_2 = [feature_2_class_1, feature_2_class_2, feature_2_class_3]
# feature_plot_3 = [feature_3_class_1, feature_3_class_2, feature_3_class_3]
# feature_plot_4 = [feature_4_class_1, feature_4_class_2, feature_4_class_3]
# feature_plot_1_text = ['Setosa', 'Versicolour', 'Veriginica']
# feature_plot_2_text = ['Setosa', 'Versicolour', 'Veriginica']
# feature_plot_3_text = ['Setosa', 'Versicolour', 'Veriginica']
# feature_plot_4_text = ['Setosa', 'Versicolour', 'Veriginica']




# # Define the features and their corresponding labels
# features = [feature_plot_1, feature_plot_2, feature_plot_3, feature_plot_4]
# feature_labels = [feature_plot_1_text, feature_plot_2_text, feature_plot_3_text, feature_plot_4_text]
# # Make list of color for each class to use in plots
# colors = ['red', 'blue', 'green']
# x_lable = ['Spetal Length', 'Spetal Width', 'Petal Length', 'Petal Width']
# # Loop through each feature
# for i, feature in enumerate(features):
#     # Create a new figure for each feature
#     plt.figure()

#     # Loop through each class and plot histogram with probability density curve
#     for j, data in enumerate(feature):
#         # Plot histogram
#         plt.hist(data, density=True, alpha=0.5, label=feature_labels[i][j], color=colors[j])

        
#         # Plot probability density curve
#         kde = gaussian_kde(data)
#         x_vals = np.linspace(min(data), max(data), 100)
#         plt.plot(x_vals, kde(x_vals), color=colors[j])

#     # Set title, labels, and legend
#     plt.title('Histogram with Probability Density Curve for ' + x_lable[i])
#     plt.xlabel(x_lable[i])
#     plt.ylabel('Number of occurences')
#     plt.legend()

# # Show all figures
plt.show()








# # Create a 3x1 grid of subplots
# fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True, sharey=True)

# # Plot histogram for class 1
# axs[0].hist(feature_plot[0], bins=20, color='red', alpha=0.5, label='Class 1')
# axs[0].set_title(feature_plot_text[0])
# axs[0].set_xlabel('Feature')
# axs[0].set_ylabel('Frequency')
# axs[0].legend()

# # Plot histogram for class 2
# axs[1].hist(feature_plot[1], bins=20, color='blue', alpha=0.5, label='Class 2')
# axs[1].set_title(feature_plot_text[0])
# axs[1].set_xlabel('Feature')
# axs[1].set_ylabel('Frequency')
# axs[1].legend()

# # Plot histogram for class 3
# axs[2].hist(feature_plot[2], bins=20, color='green', alpha=0.5, label='Class 3')
# axs[2].set_title(feature_plot_text[0])
# axs[2].set_xlabel('Feature')
# axs[2].set_ylabel('Frequency')
# axs[2].legend()

# plt.subplots_adjust(hspace=0.5)
# plt.show()




