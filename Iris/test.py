import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Make dummy confusion matrix
confusion_matrix_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Plot confusion matrix
class_labels = ['Setosa', 'Versicolour', 'Veriginica']
df_cm_test = pd.DataFrame(confusion_matrix_test, index = [i for i in class_labels], columns = [i for i in class_labels])
plt.figure(figsize = (10,7))

print("Confusion matrix for test data: ")




