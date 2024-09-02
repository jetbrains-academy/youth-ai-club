import unittest
import numpy as np

from task import Backpropagation


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class TestBackpropagation(unittest.TestCase):
    def setUp(self):
        # Initialize network with known weights and biases
        self.inputs = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        self.targets = np.array([[0.3], [0.5], [0.7], [0.9]])

        # Set initial weights and biases
        self.hidden_weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.hidden_bias = np.array([0.1, 0.2])
        self.output_weights = np.array([[0.5], [0.6]])
        self.output_bias = np.array([0.1])

        # Create instance of Backpropagation with predefined weights and biases
        self.network = Backpropagation()
        self.network.hidden_weights = self.hidden_weights
        self.network.hidden_bias = self.hidden_bias
        self.network.output_weights = self.output_weights
        self.network.output_bias = self.output_bias

    def test_train(self):
        # Train the network
        epochs = 2000
        lr = 0.1
        initial_predictions = self.network.forward(self.inputs)
        self.network.train(self.inputs, self.targets, epochs, lr)
        final_predictions = self.network.forward(self.inputs)

        # Check that loss has decreased
        initial_loss = 0.5 * np.mean((self.targets - initial_predictions) ** 2)
        final_loss = 0.5 * np.mean((self.targets - final_predictions) ** 2)
        self.assertGreater(initial_loss, final_loss, "Training did not decrease the loss")

    def test_weight_updates(self):
        # Train the network for a single epoch
        epochs = 1
        lr = 0.1
        initial_output_weights = np.copy(self.network.output_weights)
        initial_hidden_weights = np.copy(self.network.hidden_weights)

        self.network.train(self.inputs, self.targets, epochs, lr)

        # Check that weights have been updated
        self.assertFalse(np.allclose(initial_output_weights, self.network.output_weights), "Output weights did not update")
        self.assertFalse(np.allclose(initial_hidden_weights, self.network.hidden_weights), "Hidden weights did not update")

    def test_bias_updates(self):
        # Train the network for a single epoch
        epochs = 1
        lr = 0.1
        initial_output_bias = np.copy(self.network.output_bias)
        initial_hidden_bias = np.copy(self.network.hidden_bias)

        self.network.train(self.inputs, self.targets, epochs, lr)

        # Check that biases have been updated
        self.assertFalse(np.allclose(initial_output_bias, self.network.output_bias), "Output bias did not update")
        self.assertFalse(np.allclose(initial_hidden_bias, self.network.hidden_bias), "Hidden bias did not update")

if __name__ == '__main__':
    unittest.main()
