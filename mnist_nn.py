import math

class activation_funcs:
  def sigmoid(x):
    return 1 / (1 + (1 / math.e ** x))

  def relu(x):
    return max(0, x)

  def leaky_relu(x):
    return 0.01 * x if x < 0 else x

class mnist_nn:
  def __init__(self, num_hidden_layers=1, num_hidden_neurons=[1], step_size=0.5, activation_func=activation_funcs.sigmoid):
    self.num_hidden_layers = num_hidden_layers
    self.num_hidden_neurons = num_hidden_neurons
    self.step_size = step_size
    self.activation_func = activation_func
  
  def train(self):
    pass
  
  def test(self):
    pass