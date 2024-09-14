import numpy as np

from GradientDescent.Intro.main import BaseDescent, config


class Adagrad(BaseDescent):
    """
    Adaptive gradient algorithm class
    """

    def __init__(self, w0: np.ndarray, lambda_: float,
                 eps: float = config['eps_default'],
                 s0: float = config['s0_default'],
                 p: float = config['p_default']):
        """
        :param w0: weight initialization
        :param lambda_: learning rate parameter (float)
        :param eps: smoothing term (float)
        :param s0: learning rate parameter (float)
        :param p: learning rate parameter (float)
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (s0 / (s0 + k)) ** p
        self.eps = eps
        self.w = np.copy(w0)
        self.g = 0

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Changing weights with respect to gradient
        :param iteration: iteration number
        :param gradient: gradient estimate
        :return: weight difference: np.ndarray
        """
        self.g += gradient ** 2
        diff = self.eta(iteration) * gradient / (np.sqrt(self.g + self.eps))
        self.w -= diff
        return diff

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Getting objects, calculating gradient at point w
        :param X: objects' features
        :param y: objects' targets
        :return: gradient: np.ndarray
        """
        gradient = 2 * np.dot(X.T, (np.dot(X, self.w) - y)) / X.shape[0]
        return gradient
