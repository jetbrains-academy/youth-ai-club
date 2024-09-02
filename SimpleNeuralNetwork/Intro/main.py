import numpy as np


class NeuralNetworkBase:
    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_weights = np.random.rand(self.input_size, self.hidden_size)
        self.hidden_bias = np.random.rand(1, self.hidden_size)

        self.output_weights = np.random.rand(self.hidden_size, self.output_size)
        self.output_bias = np.random.rand(1, self.output_size)

    def forward(self, inputs):
        pass

    def train(self, inputs, target, epochs, lr):
        pass
