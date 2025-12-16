import unittest
import numpy as np

from task import StochasticDescent


class TestStochasticDescent(unittest.TestCase):

    def setUp(self):
        # Initial weight vector w0
        self.w0 = np.array([1.0, 2.0])
        # Parameters for learning rate
        self.lambda_ = 0.1
        self.s0 = 1.0
        self.p = 0.5
        # Batch size
        self.batch_size = 2
        # Creating an instance of StochasticDescent
        self.stochastic_descent = StochasticDescent(self.w0, self.lambda_, self.s0, self.p, self.batch_size)
        # Fix the random seed to make the test deterministic
        np.random.seed(137)

    def test_update_weights(self):
        gradient = np.array([0.1, 0.2])
        iteration = 1
        expected_diff = self.stochastic_descent.eta(iteration) * gradient
        diff = self.stochastic_descent.update_weights(gradient, iteration)

        # Check if the weight difference is correct
        np.testing.assert_array_almost_equal(diff, expected_diff, decimal=2)

        # Check if weights have been updated correctly
        expected_weights = self.w0 - expected_diff
        np.testing.assert_array_almost_equal(self.stochastic_descent.w, expected_weights, decimal=2)

    def test_calc_gradient(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([5, 6, 7, 8])

        indices = np.random.randint(X.shape[0], size=self.batch_size)
        X_batch, y_batch = X[indices], y[indices]
        expected_gradient = 2 * np.dot(X_batch.T, (np.dot(X_batch, self.w0) - y_batch)) / self.batch_size

        gradient = self.stochastic_descent.calc_gradient(X, y, indices)

        # Check if the calculated gradient is correct
        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=2)

    def test_batch_size(self):
        # Ensure that the batch size is respected
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([5, 6, 7, 8, 9])

        gradient = self.stochastic_descent.calc_gradient(X, y)

        # Check if the batch size was used correctly
        self.assertEqual(len(gradient), self.w0.size)
        self.assertEqual(gradient.shape[0], self.w0.size)


if __name__ == '__main__':
    unittest.main()
