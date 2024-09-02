import unittest
import numpy as np

from task import Adagrad


class TestAdagrad(unittest.TestCase):

    def setUp(self):
        # Initial weight vector w0
        self.w0 = np.array([1.0, 2.0])
        # Parameters for learning rate and epsilon
        self.lambda_ = 0.1
        self.eps = 1e-8
        self.s0 = 1.0
        self.p = 0.5
        # Creating an instance of Adagrad
        self.adagrad = Adagrad(self.w0, self.lambda_, self.eps, self.s0, self.p)

    def test_update_weights(self):
        gradient = np.array([0.1, 0.2])
        iteration = 1

        # Update g and calculate expected weight difference
        expected_g = gradient ** 2
        expected_diff = self.adagrad.eta(iteration) * gradient / (np.sqrt(expected_g + self.eps))

        diff = self.adagrad.update_weights(gradient, iteration)

        # Check if the g term is updated correctly
        np.testing.assert_array_almost_equal(self.adagrad.g, expected_g)

        # Check if the weight difference is correct
        np.testing.assert_array_almost_equal(diff, expected_diff)

        # Check if weights have been updated correctly
        expected_weights = self.w0 - expected_diff
        np.testing.assert_array_almost_equal(self.adagrad.w, expected_weights)

    def test_calc_gradient(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([5, 6])
        expected_gradient = 2 * np.dot(X.T, (np.dot(X, self.w0) - y)) / X.shape[0]

        gradient = self.adagrad.calc_gradient(X, y)

        # Check if the calculated gradient is correct
        np.testing.assert_array_almost_equal(gradient, expected_gradient)

    def test_initial_conditions(self):
        # Test that initial g is zero and w is correctly initialized
        self.assertEqual(self.adagrad.g, 0)
        np.testing.assert_array_equal(self.adagrad.w, self.w0)


if __name__ == '__main__':
    unittest.main()
