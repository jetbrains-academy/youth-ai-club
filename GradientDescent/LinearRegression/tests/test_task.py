import unittest
import numpy as np

from GradientDescent.FullGD.task import GradientDescent

from task import LinearRegression


class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        # Initial weight vector w0
        self.w0 = np.array([0.03, -0.01])
        # Parameters for GradientDescent
        self.lambda_ = 0.3
        self.s0 = 1.0
        self.p = 0.5
        # Fix the random seed to make the test deterministic
        np.random.seed(137)

        # Creating an instance of GradientDescent
        descent = GradientDescent(self.w0, self.lambda_, self.s0, self.p)
        # Creating an instance of LinearRegression
        self.model = LinearRegression(descent=descent, tolerance=1e-7, max_iter=1e8)

        # Sample data
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y = np.array([5, 7, 9, 11])

    def test_calc_loss(self):
        self.model.calc_loss(self.X, self.y)
        predicted_loss = np.mean((self.model.predict(self.X) - self.y) ** 2)
        # Check if the loss history has the correct loss value
        self.assertAlmostEqual(self.model.loss_history[-1], predicted_loss)

    def test_fit(self):
        # Test the fit method
        self.model.fit(self.X, self.y)
        # After fitting, the model should have a loss close to 0
        self.assertLess(self.model.loss_history[-1], 5e-2)
        # Check if the fitting process stopped due to reaching the tolerance level
        self.assertLessEqual(len(self.model.loss_history), self.model.max_iter)

    def test_predict(self):
        # Test the predict method
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        # Check if the predictions are close to the actual targets
        np.testing.assert_array_almost_equal(predictions, self.y, decimal=1)

    def test_fit_stopping(self):
        # Test if fitting stops correctly based on tolerance and max_iter
        model_low_tolerance = LinearRegression(descent=self.model.descent, tolerance=1e-2, max_iter=100)
        model_low_tolerance.fit(self.X, self.y)
        self.assertLessEqual(len(model_low_tolerance.loss_history), model_low_tolerance.max_iter)


if __name__ == '__main__':
    unittest.main()
