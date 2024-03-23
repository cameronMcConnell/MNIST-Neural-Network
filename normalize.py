import pandas as pd
from os import path

"""
This program preprocesses the MNIST dataset by normalizing pixel values
and organizing the data into training and testing sets with labels.
It saves the preprocessed datasets back to CSV files.
"""

# Define file paths for the training and testing datasets
TRAIN = path.join('./mnist_train.csv')
TEST = path.join('./mnist_test.csv')

# Read the training and testing datasets into pandas DataFrames
train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

# Extract labels from the training and testing datasets
labels_train = train.loc[:, 'label']
labels_test = test.loc[:, 'label']

# Normalize pixel values of the training and testing datasets
# by dividing each pixel value by 255 and rounding to two decimal places
norm_train = train.iloc[:, 1:].apply(lambda x: round(x / 255, 2))
norm_test = test.iloc[:, 1:].apply(lambda x: round(x / 255, 2))

# Concatenate the normalized pixel values with their respective labels
train = pd.concat([labels_train, norm_train], axis=1)
test = pd.concat([labels_test, norm_test], axis=1)

# Write the modified training and testing datasets back to CSV files
train.to_csv(TRAIN)
test.to_csv(TEST)