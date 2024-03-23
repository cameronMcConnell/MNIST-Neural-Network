import numpy as np

class MNIST_Neural_Network:
  """
  Implementation of a neural network for MNIST digit classification.

  This class represents a feedforward neural network with configurable architecture,
  activation functions, learning rate, and training parameters. It includes methods
  for training on MNIST data and testing on separate test data.

  Attributes:
  - step_size (float): Learning rate for gradient descent.
  - epochs (int): Number of training epochs.
  - hidden_layers (tuple): Tuple containing number of neurons in each hidden layer.
  - activation_function (ActivationFunctions): Activation function used in the hidden layers.
  - weights (list): List of weight matrices for each layer.
  - biases (list): List of bias vectors for each layer.

  Methods:
  - __init__: Initializes the neural network with specified parameters.
  - train: Trains the neural network using provided training data.
  - test: Tests the neural network using provided test data.
  - set_hidden_layers: Sets the number of neurons in the hidden layers.
  - __reset_parameters: Resets weights and biases to random initial values.
  - __init_weights: Initializes weights for each layer with random values.
  - __get_predicted_digit: Gets the predicted digit based on the output layer.
  - __loss: Calculates the loss between true and predicted values.
  - __forward_propagation: Performs forward propagation through the network.
  - __backward_propagation: Performs backward propagation to compute gradients.
  - __update_parameters: Updates weights and biases using computed gradients.
  """

  class ActivationFunctions:
    """Base class for activation functions."""
    def function(x):
      pass
    
    def derivative(x):
      pass

  class Sigmoid(ActivationFunctions):
    """Sigmoid activation function."""
    def __init__(self):
      self.vector = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
      self.f = None

    def function(self, x):
      self.f = self.vector(x)
      return self.f

    def derivative(self, x):
      return self.f * (1 - self.f)
  
  class Relu(ActivationFunctions):
    """Rectified Linear Unit (ReLU) activation function."""
    def function(self, x):
      return np.maximum(0, x)

    def derivative(self, x):
      return np.where(x > 0, 1, 0)
  
  class Leaky_Relu(ActivationFunctions):
    """Leaky ReLU activation function."""
    def function(self, x):
      return np.maximum(x * 0.01, x)

    def derivative(self, x):
      return np.where(x > 0, 1, 0.01)

  def __init__(self, step_size=0.01, epochs=5, hidden_layers=(10), activation_function=Sigmoid()):
    """
    Initialize the neural network with specified parameters.

    Args:
    - step_size (float): Learning rate for gradient descent.
    - epochs (int): Number of training epochs.
    - hidden_layers (tuple): Tuple containing number of neurons in each hidden layer.
    - activation_function (ActivationFunctions): Activation function to be used in the hidden layers.
    """
    self.step_size = step_size
    self.epochs = epochs
    self.activation_function = activation_function
    self.hidden_layers = hidden_layers
    self.weights = [np.random.randn(784, hidden_layers[0])]
    self.__init_weights(hidden_layers)
    self.weights.append(np.random.randn(hidden_layers[-1], 10))
    self.biases = [np.zeros((1, dim)) for dim in hidden_layers]
    self.biases.append(np.zeros((1, 10)))

  def train(self, x_train, y_train):
    """
    Train the neural network using provided training data.

    Args:
    - x_train (numpy.ndarray): Input training data.
    - y_train (numpy.ndarray): Output training data.
    """
    self.__reset_parameters()

    for epoch in range(self.epochs):
      for x, y in zip(x_train, y_train):
        activations, pre_activations = self.__forward_propagation(x)
        gradients_weights, gradients_biases = self.__backward_propagation(y, activations, pre_activations)
        self.__update_parameters(gradients_weights, gradients_biases)

      print(f"Epoch {epoch}: Loss = {self.__loss(y, activations[-1])}")
        
  def test(self, x_test):
    """
    Test the neural network using provided test data.

    Args:
    - x_test (numpy.ndarray): Input test data.

    Returns:
    - y_pred (list): Predicted labels for the test data.
    """
    y_pred = []

    for x in x_test:
      activations, _ = self.__forward_propagation(x)
      predicted_digit = self.__get_predicted_digit(activations[-1])
      y_pred.append(predicted_digit)
    
    return y_pred

  def set_hidden_layers(self, hidden_layers):
    """
    Set the number of neurons in the hidden layers.

    Args:
    - hidden_layers (tuple): Tuple containing number of neurons in each hidden layer.
    """
    self.weights = [np.random.randn(784, hidden_layers[0])]
    self.__init_weights(hidden_layers)
    self.weights.append(np.random.randn(hidden_layers[-1], 10))
    self.biases = [np.zeros((1, dim)) for dim in hidden_layers]
    self.biases.append(np.zeros((1, 10)))

  def __reset_parameters(self):
    """
    Reset weights and biases to random initial values.
    """
    self.weights = [np.random.randn(784, self.hidden_layers[0])]
    self.__init_weights(self.hidden_layers)
    self.weights.append(np.random.randn(self.hidden_layers[-1], 10))
    self.biases = [np.zeros((1, dim)) for dim in self.hidden_layers]
    self.biases.append(np.zeros((1, 10)))

  def __init_weights(self, dims):
    """
    Initialize weights for each layer with random values.

    Args:
    - dims (tuple): Tuple containing number of neurons in each hidden layer.
    """
    for i in range(1, len(dims) - 1):
      self.weights.append(np.random.randn(dims[i], dims[i + 1]))

  def __get_predicted_digit(self, output_layer):
    """
    Get the predicted digit based on the output layer.

    Args:
    - output_layer (numpy.ndarray): Output layer of the neural network.

    Returns:
    - digit (int): Predicted digit.
    """
    max_num = digit = 0

    for i, num in enumerate(output_layer[0]):
      if num > max_num:
        num = max_num
        digit = i
    
    return digit
  
  def __loss(self, y_true, y_pred):
    """
    Calculate the loss between true and predicted values.

    Args:
    - y_true (numpy.ndarray): True output values.
    - y_pred (numpy.ndarray): Predicted output values.

    Returns:
    - loss (float): Loss value.
    """
    return 0.5 * np.sum((y_true - y_pred) ** 2)

  def __forward_propagation(self, x):
    """
    Perform forward propagation through the network.

    Args:
    - x (numpy.ndarray): Input data.

    Returns:
    - activations (list): List of activation values for each layer.
    - pre_activations (list): List of pre-activation values for each layer.
    """
    activations, pre_activations = [np.reshape(x, (1, 784))], []

    for i in range(len(self.weights)):
      pre_activation = np.dot(activations[-1], self.weights[i]) + self.biases[i]
      activation = self.activation_function.function(pre_activation)
      activations.append(activation)
      pre_activations.append(pre_activation)
    
    return activations, pre_activations

  def __backward_propagation(self, y, activations, pre_activations):
    """
    Perform backward propagation to compute gradients.

    Args:
    - y (numpy.ndarray): True output values.
    - activations (list): List of activation values for each layer.
    - pre_activations (list): List of pre-activation values for each layer.

    Returns:
    - gradients_weights (deque): Deque of gradients of weights.
    - gradients_biases (deque): Deque of gradients of biases.
    """
    gradients_weights, gradients_biases = [], []

    error = activations[-1] - y
    delta = error * self.activation_function.derivative(pre_activations[-1])
    gradient_weights = np.dot(activations[-2].T, delta)
    gradient_biases = np.sum(delta, axis=0)
    gradients_weights.appendleft(gradient_weights)
    gradients_biases.appendleft(gradient_biases)

    for i in range(len(self.weights) - 2, -1, -1):
      error = np.dot(delta, self.weights[i+1].T)
      delta = error * self.activation_function.derivative(pre_activations[i])
      gradient_weights = np.dot(activations[i].T, delta)
      gradient_biases = np.sum(delta, axis=0)
      gradients_weights.append(gradient_weights)
      gradients_biases.append(gradient_biases)

    return gradients_weights[::-1], gradients_biases[::-1]

  def __update_parameters(self, gradients_weights, gradients_biases):
    """
    Update weights and biases using gradients computed during backpropagation.

    Args:
    - gradients_weights (deque): Deque of gradients of weights.
    - gradients_biases (deque): Deque of gradients of biases.
    """
    for i in range(len(self.weights)):
      self.weights[i] -= self.step_size * gradients_weights[i]
      self.biases[i] -= self.step_size * gradients_biases[i]