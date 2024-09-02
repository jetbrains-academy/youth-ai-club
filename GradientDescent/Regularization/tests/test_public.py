import unittest
import numpy as np

from task import AdagradReg


class TestAdagradReg(unittest.TestCase):

    def setUp(self):
        # Initial weight vector w0
        self.w0 = np.array([1.0, 2.0])
        # Parameters for learning rate, epsilon, and regularization
        self.lambda_ = 0.1
        self.eps = 1e-8
        self.mu = 0.01
        self.s0 = 1.0
        self.p = 0.5
        # Creating an instance of AdagradReg
        self.adagrad_reg = AdagradReg(self.w0, self.lambda_, self.eps, self.mu, self.s0, self.p)

    def test_calc_gradient(self):
        # Test the calc_gradient method with L2 regularization
        X = np.array([[1, 2], [3, 4]])
        y = np.array([5, 6])
        base_gradient = 2 * np.dot(X.T, (np.dot(X, self.w0) - y)) / X.shape[0]
        l2_gradient = self.mu * self.w0
        expected_gradient = base_gradient + l2_gradient

        gradient = self.adagrad_reg.calc_gradient(X, y)

        # Check if the calculated gradient is correct, including the L2 term
        np.testing.assert_array_almost_equal(gradient, expected_gradient)

if __name__ == '__main__':
    unittest.main()
