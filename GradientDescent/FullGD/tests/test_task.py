import unittest
import numpy as np

from task import GradientDescent


class TestGradientDescent(unittest.TestCase):

    def setUp(self):
        # Initial weight vector w0
        self.w0 = np.array([1.0, 2.0])
        # Parameters for learning rate
        self.lambda_ = 0.1
        self.s0 = 1.0
        self.p = 0.5
        # Creating an instance of GradientDescent
        self.gradient_descent = GradientDescent(self.w0, self.lambda_, self.s0, self.p)

    def test_update_weights(self):
        gradient = np.array([0.1, 0.2])
        iteration = 1
        expected_diff = self.gradient_descent.eta(iteration) * gradient
        diff = self.gradient_descent.update_weights(gradient, iteration)

        # Check if the weight difference is correct
        np.testing.assert_array_almost_equal(diff, expected_diff, decimal=2)

        # Check if weights have been updated correctly
        expected_weights = self.w0 - expected_diff
        np.testing.assert_array_almost_equal(self.gradient_descent.w, expected_weights, decimal=2)

    def test_calc_gradient(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([5, 6])
        expected_gradient = 2 * np.dot(X.T, (np.dot(X, self.w0) - y)) / X.shape[0]

        gradient = self.gradient_descent.calc_gradient(X, y)

        # Check if the calculated gradient is correct
        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=2)


if __name__ == '__main__':
    unittest.main()
