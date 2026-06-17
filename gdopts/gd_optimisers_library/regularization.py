from abc import ABC, abstractmethod
import numpy as np


class Regularizer(ABC):
    @abstractmethod
    def apply(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        pass


class NoRegularization(Regularizer):
    def apply(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad


class L2Regularizer(Regularizer):
    def __init__(self, lambda_: float = 1e-4):
        self.lambda_ = float(lambda_)

    def apply(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad + self.lambda_ * x


class L1Regularizer(Regularizer):
    def __init__(self, lambda_: float = 1e-4):
        self.lambda_ = float(lambda_)

    def apply(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad + self.lambda_ * np.sign(x)


class ElasticNetRegularizer(Regularizer):
    def __init__(self, lambda1: float = 1e-4, lambda2: float = 1e-4):
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)

    def apply(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad + self.lambda1 * np.sign(x) + self.lambda2 * x