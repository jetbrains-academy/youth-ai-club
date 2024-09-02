import numpy as np

from SimpleNeuralNetwork.Intro.main import NeuralNetworkBase


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Forward(NeuralNetworkBase):
    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        super().__init__(input_size, hidden_size, output_size)
        self.hidden_outputs = None
        self.predicted_outputs = None

    def forward(self, inputs):
        self.hidden_outputs = sigmoid(inputs.dot(self.hidden_weights) + self.hidden_bias)
        self.predicted_outputs = sigmoid(self.hidden_outputs.dot(self.output_weights) + self.output_bias)
        return self.predicted_outputs
