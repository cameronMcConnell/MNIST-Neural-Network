from sklearn.model_selection import train_test_split
from os import path
import pandas as pd
import numpy as np

class Data():

  def __init__(self):
    self.test_path = path.join('./mnist_test.csv')
    self.train_path = path.join('./mnist_train.csv')
    (self.x_train, self.y_train), (self.x_test, self.y_test) = self.__load_data()
    
  def shuffle_data(self, test_size):
    test, train = self.__read_data()

    data = pd.concat([test, train]).sample(frac=1)
    train, test = train_test_split(data, test_size=test_size)

    self.x_train, self.y_train = self.__split_data(train, 1)
    self.x_test, self.y_test = self.__split_data(test, 0)

  def get_data(self, dtype):
    data = {'train': [self.x_train, self.y_train], 'test': [self.x_test, self.y_test]}
    return data[dtype]

  def __read_data(self):
    return pd.read_csv(self.test_path), pd.read_csv(self.train_path)
    
  def __load_data(self):
    test, train = self.__read_data()
    return self.__split_data(train, 1), self.__split_data(test, 0)
    
  def __split_data(self, data, flag):
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