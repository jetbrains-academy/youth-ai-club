import unittest
import numpy as np

from task import Forward


class TestForward(unittest.TestCase):

    def setUp(self):
        # Set up the neural network with known weights and biases
        self.inputs = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.hidden_weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.hidden_bias = np.array([0.1, 0.2])
        self.output_weights = np.array([[0.5], [0.6]])
        self.output_bias = np.array([0.1])

        # Create an instance of Forward with predefined weights and biases
        self.network = Forward()
        self.network.hidden_weights = self.hidden_weights
        self.network.hidden_bias = self.hidden_bias
        self.network.output_weights = self.output_weights
        self.network.output_bias = self.output_bias

    def test_forward(self):
        # Expected output calculated manually or with known correct implementation
        expected_hidden_outputs = 1 / (1 + np.exp(- (self.inputs.dot(self.hidden_weights) + self.hidden_bias)))
        expected_predicted_output = 1 / (
                    1 + np.exp(- (expected_hidden_outputs.dot(self.output_weights) + self.output_bias)))

        # Run the forward method
        predicted_output = self.network.forward(self.inputs)

        # Assert the output is as expected
        np.testing.assert_array_almost_equal(predicted_output, expected_predicted_output)


if __name__ == '__main__':
    unittest.main()
