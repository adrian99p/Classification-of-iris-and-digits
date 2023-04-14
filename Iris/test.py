import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make dummy confusion matrix
confusion_matrix = np.array([[0.9, 0.1, 0.0],
                                [0.1, 0.8, 0.1],
                                [0.0, 0.1, 0.9]])

# Make seaborn heatmap
sns.set(font_scale=1.4)
sns.heatmap(confusion_matrix, annot=True, annot_kws={"size": 16}, fmt="g", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels,
            cbar=False)




