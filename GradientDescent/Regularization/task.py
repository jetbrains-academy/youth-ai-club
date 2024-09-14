import numpy as np

from GradientDescent.Intro.main import config
from GradientDescent.Adagrad.task import Adagrad


class AdagradReg(Adagrad):
    """
    Adaptive gradient algorithm with regularization class
    """

    def __init__(self, w0: np.ndarray, lambda_: float,
                 eps: float = config['eps_default'],
                 mu: float = config['mu_default'],
                 s0: float = config['s0_default'],
                 p: float = config['p_default']):
        """
        :param mu: l2 coefficient
        """
        super().__init__(w0=w0, lambda_=lambda_, eps=eps, s0=s0, p=p)
        self.mu = mu

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        return super().update_weights(gradient, iteration)

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        l2 = self.w * self.mu
        return super().calc_gradient(X, y) + l2
