import pandas as pd
import os

TRAIN = os.path.join('./mnist_train.csv')
TEST = os.path.join('./mnist_test.csv')

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

labels_train = train.loc[:, 'label']
labels_test = test.loc[:, 'label']

norm_train = train.iloc[:, 1:].apply(lambda x: round(x / 255, 2))
norm_test = test.iloc[:, 1:].apply(lambda x: round(x / 255, 2))

train = pd.concat([labels_train, norm_train], axis=1)
test = pd.concat([labels_test, norm_test], axis=1)

train.to_csv(TRAIN)
test.to_csv(TEST)