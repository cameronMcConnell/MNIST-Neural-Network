from sklearn.model_selection import train_test_split
from os import path
import pandas as pd
import numpy as np

class Data:
  """
  Utility class for loading and processing MNIST dataset.

  This class provides methods for loading, shuffling, and splitting the MNIST dataset
  into training and test sets. It also includes methods for retrieving data of specific types.

  Attributes:
  - test_path (str): Path to the test dataset file.
  - train_path (str): Path to the train dataset file.
  - x_train (numpy.ndarray): Features of the training dataset.
  - y_train (numpy.ndarray): Labels of the training dataset.
  - x_test (numpy.ndarray): Features of the test dataset.
  - y_test (numpy.ndarray): Labels of the test dataset.

  Methods:
  - __init__: Initializes the dataset by loading and splitting the data.
  - shuffle_data: Shuffles the dataset and updates training and test sets accordingly.
  - get_data: Retrieves data of specified type (train/test).
  - __read_data: Reads the data from CSV files.
  - __load_data: Loads and splits the data into features and labels.
  - __split_data: Splits the data into features and labels.
  """

  def __init__(self):
    """
    Initializes the dataset by loading and splitting the data.

    This method sets the paths for the test and train dataset files and loads
    the data into memory. It then splits the data into features and labels for
    both training and testing sets.
    """
    self.test_path = path.join('./mnist_test.csv')
    self.train_path = path.join('./mnist_train.csv')
    (self.x_train, self.y_train), (self.x_test, self.y_test) = self.__load_data()
    
  def shuffle_data(self, test_size):
    """
    Shuffles the dataset and updates training and test sets.

    Args:
    - test_size (float): Proportion of the dataset to include in the test split.
    """
    test, train = self.__read_data()

    data = pd.concat([test, train]).sample(frac=1)
    train, test = train_test_split(data, test_size=test_size)

    self.x_train, self.y_train = self.__split_data(train, 1)
    self.x_test, self.y_test = self.__split_data(test, 0)

  def get_data(self, dtype):
    """
    Retrieves data of specified type (train/test).

    Args:
    - dtype (str): Type of data to retrieve ('train' or 'test').

    Returns:
    - numpy.ndarray: Features of the specified data type.
    - numpy.ndarray: Labels of the specified data type.
    """
    data = {'train': [self.x_train, self.y_train], 'test': [self.x_test, self.y_test]}
    
    try:
      return data[dtype]
    except:
      raise ValueError("Invalid data type. Use 'train' or 'test'.")

  def __read_data(self):
    """Reads the data from CSV files."""
    return pd.read_csv(self.test_path), pd.read_csv(self.train_path)
    
  def __load_data(self):
    """Loads and splits the data into features and labels."""
    test, train = self.__read_data()
    return self.__split_data(train, True), self.__split_data(test, False)
    
  def __split_data(self, data, flag):
    """
    Splits the data into features and labels.

    Args:
    - data (pd.DataFrame): DataFrame containing the dataset.
    - flag (bool): Flag to determine whether to encode labels.

    Returns:
    - numpy.ndarray: Features of the dataset.
    - numpy.ndarray: Labels of the dataset.
    """
    x = data.iloc[:, 2:].to_numpy()

    labels = data.loc[:, 'label']

    if flag:
      y = []

      for i, num in enumerate(labels):
        y.append(np.zeros((1, 10)))
        y[i][0][num] = 1
    else:
      y = labels

    y = np.array(y)

    return x, y