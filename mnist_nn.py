import numpy as np

class mnist_nn:
  def __init__(self, step_size=0.5):
    self.step_size = step_size
    self.biases = 2 * np.random.rand(3, 1) - 1
    self.hidden_layers = [np.array([]), np.array([])]
    self.weights = [2 * np.random.rand(784, 16) - 1,
                    2 * np.random.rand(16, 16) - 1,
                    2 * np.random.rand(16, 10) - 1]
  
  def train(self, x, y):
    test = np.random.rand(1, 784)
    relu = np.vectorize(lambda x: max(0, x))

    self.hidden_layers[0] = relu(test.dot(self.weights[0]) + self.biases[0])

    self.hidden_layers[1] = relu(self.hidden_layers[0].dot(self.weights[1]) + self.biases[1])

    output = relu(self.hidden_layers[1].dot(self.weights[2]) + self.biases[2])

  def test(self, x):
    pass

nn = mnist_nn()
nn.train(1, 2)
print(nn.hidden_layers)