import unittest
import numpy as np

from task import MomentumDescent


class TestMomentumDescent(unittest.TestCase):

    def setUp(self):
        # Initial weight vector w0
        self.w0 = np.array([1.0, 2.0])
        # Parameters for learning rate and momentum
        self.lambda_ = 0.1
        self.alpha = 0.9
        self.s0 = 1.0
        self.p = 0.5
        # Creating an instance of MomentumDescent
        self.momentum_descent = MomentumDescent(self.w0, self.lambda_, self.alpha, self.s0, self.p)

    def test_update_weights(self):
        gradient = np.array([0.1, 0.2])
        iteration = 1

        # Expected difference calculation
        expected_h = self.alpha * 0 + self.momentum_descent.eta(iteration) * gradient
        diff = self.momentum_descent.update_weights(gradient, iteration)

        # Check if the h (momentum) term is correct
        np.testing.assert_array_almost_equal(diff, expected_h, decimal=2)

        # Check if weights have been updated correctly
        expected_weights = self.w0 - expected_h
        np.testing.assert_array_almost_equal(self.momentum_descent.w, expected_weights, decimal=2)

        # Test the update again to ensure momentum accumulates correctly
        gradient_new = np.array([0.2, 0.1])
        iteration += 1
        expected_h = self.alpha * expected_h + self.momentum_descent.eta(iteration) * gradient_new
        diff = self.momentum_descent.update_weights(gradient_new, iteration)

        # Check if the h (momentum) term is correct after accumulation
        np.testing.assert_array_almost_equal(diff, expected_h, decimal=2)

        # Check if weights have been updated correctly again
        expected_weights -= expected_h
        np.testing.assert_array_almost_equal(self.momentum_descent.w, expected_weights, decimal=2)

    def test_calc_gradient(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([5, 6])
        expected_gradient = 2 * np.dot(X.T, (np.dot(X, self.w0) - y)) / X.shape[0]

        gradient = self.momentum_descent.calc_gradient(X, y)

        # Check if the calculated gradient is correct
        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=2)

    def test_initial_conditions(self):
        # Test that initial h is zero and w is correctly initialized
        self.assertEqual(self.momentum_descent.h, 0)
        np.testing.assert_array_equal(self.momentum_descent.w, self.w0)


if __name__ == '__main__':
    unittest.main()
