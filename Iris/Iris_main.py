import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data from the csv file and split into training and testing
training_data = pd.read_csv('class_1.csv')
testing_data = pd.read_csv('class_2.csv')