# MNIST-Neural-Network

This repository contains Python scripts for preprocessing the MNIST dataset, implementing a neural network for digit classification, and testing the neural network's performance.

## Files

### 1. `data.py`

This Python script contains the implementation of the `Data` class, which is responsible for loading, shuffling, and splitting the MNIST dataset into training and testing sets. The `Data` class also provides methods for retrieving data and shuffling the dataset.

### 2. `mnist_nn.py`

This Python script contains the implementation of the `MNIST_Neural_Network` class, which represents a feedforward neural network for digit classification. It includes methods for training on MNIST data, testing on separate test data, and saving/loading models. The neural network architecture, hyperparameters, and activation functions can be customized through this script.

### 3. `normalize.py`

This Python script contains functions for normalizing the pixel values of the MNIST dataset. Normalization is a common preprocessing step in machine learning to scale feature values to a uniform range, which can improve model convergence and performance.

### 4. `tester.py`

This Python script contains the implementation of the `MNIST_Neural_Network_Tester` class, which provides methods for training, testing, and evaluating neural network models using the MNIST dataset. The tester class allows users to adjust hyperparameters, shuffle data, save/load models, and calculate accuracy.

### 5. `mnist_data.zip`

This ZIP archive contains the CSV files `mnist_train.csv` and `mnist_test.csv`, which are preprocessed versions of the MNIST dataset after running the provided scripts. The CSV files contain normalized pixel values along with corresponding labels for training and testing.

## Dataset

You can download the original MNIST dataset from the following link:

[Download MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download)

## Dependencies

The following Python libraries are required to run the scripts:

- `pandas`: For data manipulation and handling CSV files.
- `numpy`: For numerical computations and array operations.
- `scikit-learn`: For train-test split functionality.
- `matplotlib`: For data visualization (optional).

You can install the dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage

To use these scripts, follow these steps:

### 1. Clone this repository:

```bash
git clone https://github.com/cameronMcConnell/MNIST-Neural-Network.git
```

### 2. Navigate to the clones repository:

```bash
cd MNIST-Neural-Network
```

### 3. Unzip the `mnist_data.zip` file to extract the data set files:

```bash
unzip mnist_data.zip
```

## Example Usage

### 1. Loading the dataset:

```python
from data import Data

# Create an instance of the Data class
data = Data()

# Get the training and testing data
x_train, y_train = data.get_data('train')
x_test, y_test = data.get_data('test')
```

### 2. Training the neural network:

```python
from mnist_nn import MNIST_Neural_Network

# Create an instance of the neural network
nn = MNIST_Neural_Network()

# Train the neural network
nn.train(x_train, y_train)
```

### 3. Testing the neural network:

```python
from tester import MNIST_Neural_Network_Tester

# Create an instance of the tester
tester = MNIST_Neural_Network_Tester(nn)

# Test the neural network
tester.test()
```

## Contributors

* [Cameron McConnell](https://www.linkedin.com/in/cameron-mcconnell-704b17225/)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
