import numpy as np

from GradientDescent.Intro.main import BaseDescent, config


class StochasticDescent(BaseDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, w0: np.ndarray, lambda_: float,
                 s0: float = config['s0_default'],
                 p: float = config['p_default'],
                 batch_size: int = config['batch_size_default']):
        """
        :param w0: weight initialization
        :param lambda_: learning rate parameter (float)
        :param s0: learning rate parameter (float)
        :param p: learning rate parameter (float)
        :param batch_size: batch size (int)
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (s0 / (s0 + k)) ** p
        self.batch_size = batch_size
        self.w = np.copy(w0)

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Changing weights with respect to gradient
        :param iteration: iteration number
        :param gradient: gradient estimate
        :return: weight difference: np.ndarray
        """
        diff = self.eta(iteration) * gradient
        self.w -= diff
        return diff

    def calc_gradient(self, X: np.ndarray, y: np.ndarray, indices=None) -> np.ndarray:
        """
        Getting objects, calculating gradient at point w
        :param X: objects' features
        :param y: objects' targets
        :param indices: for test purposes
        :return: gradient: np.ndarray
        """
        if indices is None:
            indices = np.random.randint(X.shape[0], size=self.batch_size)
        X, y = X[indices], y[indices]
        gradient = 2 * np.dot(X.T, (np.dot(X, self.w) - y)) / self.batch_size
        return gradient
