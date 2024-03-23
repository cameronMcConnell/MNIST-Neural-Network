from data import Data
import pickle as pkl

class MNIST_Neural_Network_Tester:
  """
  Utility class for testing and evaluating neural network models.

  This class provides methods for training, testing, and evaluating neural network models
  using the MNIST dataset. It also offers functionalities for adjusting hyperparameters,
  shuffling data, saving/loading models, and calculating accuracy.

  Attributes:
  - nn (MNIST_Neural_Network): Neural network model to be tested.
  - data (Data): MNIST dataset used for training and testing.

  Methods:
  - __init__: Initializes the tester with a neural network model and MNIST dataset.
  - train: Trains the neural network model using training data from the MNIST dataset.
  - test: Tests the neural network model using test data from the MNIST dataset.
  - set_epochs: Sets the number of epochs for training.
  - set_hidden_layers: Sets the number of neurons in the hidden layers of the model.
  - set_step_size: Sets the learning rate for training.
  - set_activation_function: Sets the activation function for the hidden layers of the model.
  - shuffle_data: Shuffles the training and test data.
  - save_neural_net: Saves the neural network model to a file using pickle.
  - load_neural_net: Loads a neural network model from a file using pickle.
  - __calculate_accuracy: Calculates and prints the accuracy of the model on the test data.
  """

  def __init__(self, nn):
    """
    Initialize the tester with a neural network model.

    Args:
    - nn: Neural network model to be tested.
    """
    self.nn = nn
    self.data = Data()

  def train(self):
    """Train the neural network model using training data."""
    self.nn.train(self.data.x_train, self.data.y_train)
  
  def test(self):
    """Test the neural network model using test data."""
    y_pred = self.nn.test(self.data.x_test)
    self.__calculate_accuracy(y_pred)
  
  def set_epochs(self, epochs):
    """
    Set the number of epochs for training the neural network.
    
    Args:
    - epochs (int): Number of epochs.
    """
    self.nn.epochs = epochs
  
  def set_hidden_layers(self, hidden_layers):
    """
    Set the configuration of hidden layers in the neural network.
    
    Args:
    - hidden_layers (tuple): Tuple containing number of neurons in each hidden layer.
    """
    self.nn.set_hidden_layers(hidden_layers)
  
  def set_step_size(self, step_size):
    """
    Set the step size (learning rate) for the neural network.
    
    Args:
    - step_size (float): Step size for gradient descent.
    """
    self.nn.step_size = step_size

  def set_activation_function(self, activation_function):
    """
    Set the activation function for the neural network.
    
    Args:
    - activation_function (str): Name of the activation function.
    """
    functions = {'Sigmoid': self.nn.Sigmoid(), 'Relu': self.nn.Relu(), 'Leaky_Relu': self.nn.Leaky_Relu()}
    self.nn.activation_function = functions[activation_function]

  def shuffle_data(self, test_size):
    """
    Shuffle the data used for training and testing.

    Args:
    - test_size (float): Size of the test data as a proportion of the total data.
    """
    self.data.shuffle_data(test_size)

  def save_neural_net(self, model_name):
    """
    Save the neural network model to a file.

    Args:
    - model_name (str): Name of the file to save the model.
    """
    with open(f'./{model_name}.pkl', 'wb') as f:
      pkl.dump(self.data, f)

  def load_neural_net(self, model_name):
    """
    Load a neural network model from a file.

    Args:
    - model_name (str): Name of the file containing the saved model.
    """
    try:
      with open(f'./{model_name}.pkl', 'rb') as f:
        self.nn = pkl.load(f)
    except:
      print(f'Error, no model with the name {model_name}.')
  
  def __calculate_accuracy(self, y_pred):
    """
    Calculate and print the accuracy of the neural network model.

    Args:
    - y_pred (list): Predicted labels for the test data.
    """
    total = len(self.data.y_test)
    correct = 0

    for y_p, y_t in zip(y_pred, self.data.y_test):
      correct += y_p == y_t

    print(f"Accuracy = {correct / total}")