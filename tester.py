from data import Data
import pickle as pkl

class Neural_Net_Tester:

  def __init__(self, nn):
    self.nn = nn
    self.data = Data()

  def train(self):
    self.nn.train(self.data.x_train, self.data.y_train)
  
  def test(self):
    y_pred = self.nn.test(self.data.x_test)
    self.__calculate_accuracy(y_pred)
  
  def set_epochs(self, epochs):
    self.nn.epochs = epochs
  
  def set_hidden_layers(self, hidden_layers):
    self.nn.set_hidden_layers(hidden_layers)
  
  def set_step_size(self, step_size):
    self.nn.step_size = step_size

  def set_activation_function(self, activation_function):
    functions = {'Sigmoid': self.nn.Sigmoid(), 'Relu': self.nn.Relu(), 'Leaky_Relu': self.nn.Leaky_Relu()}
    self.nn.activation_function = functions[activation_function]

  def shuffle_data(self, test_size):
    self.data.shuffle_data(test_size)

  def save_neural_net(self, model_name):
    with open(f'./{model_name}.pkl', 'wb') as f:
      pkl.dump(self.data, f)

  def load_neural_net(self, model_name):
    try:
      with open(f'./{model_name}.pkl', 'rb') as f:
        self.nn = pkl.load(f)
    except:
      print(f'Error, no model with the name {model_name}.')
  
  def __calculate_accuracy(self, y_pred):
    total = len(self.data.y_test)
    correct = 0

    for y_p, y_t in zip(y_pred, self.data.y_test):
      correct += y_p == y_t

    print(f"Accuracy = {correct / total}")