import numpy as np

from GradientDescent.Intro.main import BaseDescent, config


class LinearRegression:
    """
    Linear regression class
    """

    def __init__(self, descent: BaseDescent,
                 tolerance: float = config['tolerance_default'],
                 max_iter: int = config['max_iter_default']):
        """
        :param descent: Descent class
        :param tolerance: float stopping criterion for square of Euclidean norm of a weight difference
        :param max_iter: int stopping criterion for iterations
        """
        self.descent = descent
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_history = []

    def calc_loss(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Getting objects, calculating loss
        :param X: objects' features
        :param y: objects' target
        """
        loss = np.mean((self.predict(X) - y) ** 2)
        self.loss_history.append(loss)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Getting objects, fitting descent weights
        :param X: objects' features
        :param y: objects' target
        :return: self
        """
        n_iter = 0
        while n_iter < self.max_iter:
            self.calc_loss(X, y)
            diff = self.descent.step(X, y, iteration=n_iter)
            n_iter += 1
            w_diff = np.linalg.norm(diff) ** 2
            if w_diff < self.tolerance:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Getting objects, predicting targets
        :param X: objects' features
        :return: predicted targets
        """
        return np.dot(X, self.descent.w)
