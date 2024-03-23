from collections import deque
import numpy as np

class MNIST_Neural_Network:

  class activation_functions:
    def function(x):
      pass
    
    def derivative(x):
      pass

  class Sigmoid(activation_functions):
    def __init__(self):
      self.vector = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
      self.f = None

    def function(self, x):
      self.f = self.vector(x)
      return self.f

    def derivative(self, x):
      return self.f * (1 - self.f)
  
  class Relu(activation_functions):
    def function(self, x):
      return np.maximum(0, x)

    def derivative(self, x):
      return np.where(x > 0, 1, 0)
  
  class Leaky_Relu(activation_functions):
    def function(self, x):
      return np.maximum(x * 0.01, x)

    def derivative(self, x):
      return np.where(x > 0, 1, 0.01)

  def __init__(self, step_size=0.01, epochs=5, hidden_layers=(10), activation_function=Sigmoid()):
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
    self.__reset_parameters()

    for epoch in range(self.epochs):
      for x, y in zip(x_train, y_train):
        activations, pre_activations = self.__forward_propagation(x)

        gradients_weights, gradients_biases = self.__backward_propagation(y, activations, pre_activations)

        self.__update_parameters(gradients_weights, gradients_biases)

      print(f"Epoch {epoch}: Loss = {self.__loss(y, activations[-1])}")
        
  def test(self, x_test):
    y_pred = []

    for x in x_test:
      activations, _ = self.__forward_propagation(x)
      predicted_digit = self.__get_predicted_digit(activations[-1])
      y_pred.append(predicted_digit)
    
    return y_pred

  def set_hidden_layers(self, hidden_layers):
    self.weights = [np.random.randn(784, hidden_layers[0])]

    self.__init_weights(hidden_layers)

    self.weights.append(np.random.randn(hidden_layers[-1], 10))
    
    self.biases = [np.zeros((1, dim)) for dim in hidden_layers]

    self.biases.append(np.zeros((1, 10)))

  def __reset_parameters(self):
    self.weights = [np.random.randn(784, self.hidden_layers[0])]

    self.__init_weights(self.hidden_layers)

    self.weights.append(np.random.randn(self.hidden_layers[-1], 10))
    
    self.biases = [np.zeros((1, dim)) for dim in self.hidden_layers]

    self.biases.append(np.zeros((1, 10)))

  def __init_weights(self, dims):
    for i in range(1, len(dims) - 1):
      self.weights.append(np.random.randn(dims[i], dims[i + 1]))

  def __get_predicted_digit(self, output_layer):
    max_num = digit = 0

    for i, num in enumerate(output_layer[0]):
      if num > max_num:
        num = max_num
        digit = i
    
    return digit
  
  def __loss(self, y_true, y_pred):
    return 0.5 * np.sum((y_true - y_pred) ** 2)

  def __forward_propagation(self, x):
    activations, pre_activations = [np.reshape(x, (1, 784))], []

    for i in range(len(self.weights)):
      pre_activation = np.dot(activations[-1], self.weights[i]) + self.biases[i]
      activation = self.activation_function.function(pre_activation)
      activations.append(activation)
      pre_activations.append(pre_activation)
    
    return activations, pre_activations

  def __backward_propagation(self, y, activations, pre_activations):
    gradients_weights, gradients_biases = deque(), deque()

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
      gradients_weights.appendleft(gradient_weights)
      gradients_biases.appendleft(gradient_biases)
    
    return gradients_weights, gradients_biases

  def __update_parameters(self, gradients_weights, gradients_biases):
    for i in range(len(self.weights)):
      self.weights[i] -= self.step_size * gradients_weights[i]
      self.biases[i] -= self.step_size * gradients_biases[i]