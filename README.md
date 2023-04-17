# Classification-of-iris-and-digits
TTT4275 Project 

This project focuses on classifying Irises based on their features and handwritten digits from the MNIST dataset using the k-nearest neighbors (KNN) algorithm.

Overview
The main goal of this project is to develop a machine learning model that can accurately classify Iris flowers and handwritten digits. The project utilizes the popular Iris dataset, which contains 150 samples of Iris flowers, each with four features (sepal length, sepal width, petal length, and petal width) and three class labels (setosa, versicolor, and virginica). Additionally, the project uses the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).

Project Components
The project is organized into the following components:

Data Preprocessing: The Iris dataset and MNIST dataset are loaded and preprocessed. For the Iris dataset, the features are scaled to ensure that they are on the same scale. For the MNIST dataset, the images are normalized to grayscale values between 0 and 1.

Feature Extraction: The relevant features from the Iris dataset (sepal length, sepal width, petal length, and petal width) are extracted as input features for the KNN algorithm. For the MNIST dataset, the images are flattened into one-dimensional arrays of pixel values to serve as input features for the KNN algorithm.

KNN Classification: The KNN algorithm is used to train a classification model on the preprocessed Iris and MNIST datasets. The trained model can then be used to predict the class labels of new, unseen samples.

Evaluation: The performance of the trained model is evaluated using appropriate evaluation metrics, such as accuracy, precision, recall, and F1 score. The evaluation results are analyzed to assess the accuracy and effectiveness of the classification model.

Visualization: The results of the classification model can be visualized using various techniques, such as confusion matrices, scatter plots, and bar charts, to gain insights into the model's performance and make interpretations.

Requirements
The project requires the following dependencies:

Python 3.x
NumPy
Scikit-learn
Matplotlib
Usage
Clone the project repository to your local machine.
Install the required dependencies using pip or conda.
Run the main script or Jupyter Notebook to preprocess the datasets, train the KNN classification model, and evaluate its performance.
Visualize the results using the provided visualization techniques.
Modify the code and experiment with different parameters or techniques to improve the classification performance, if desired.
License
This project is released under the MIT License, which allows for free use, modification, and distribution of the code for both personal and commercial purposes.

Acknowledgements
The Iris dataset used in this project is sourced from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/iris).
The MNIST dataset used in this project is sourced from the MNIST Database (http://yann.lecun.com/exdb/mnist/).
Conclusion
This project provides a comprehensive implementation of classifying Irises and handwritten digits using the KNN algorithm. It serves as a starting point for further exploration and experimentation with other machine learning techniques or datasets. Feel free to customize and extend the project according to your specific requirements and goals.
