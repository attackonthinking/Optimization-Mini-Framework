
from numpy.typing import NDArray
import numpy as np

from .objective import Objective

class MSE(Objective):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__()
        self.X = np.c_[np.ones(X.shape[0]), X]
        self.y = y.astype(np.float64)
        self.tags = {"regression", "linear-regression", "convex"}

    def get_n_samples(self) -> int:
        return len(self.y)

    def _compute_value(self, w: NDArray[np.float64], indices=None) -> float:
        X_b = self.X[indices] if indices is not None else self.X
        y_b = self.y[indices] if indices is not None else self.y

        preds = X_b @ w
        errors = preds - y_b
        return float(np.mean(errors ** 2))

    def _compute_grad(self, w: NDArray[np.float64], indices=None) -> NDArray[np.float64]:
        X_b = self.X[indices] if indices is not None else self.X
        y_b = self.y[indices] if indices is not None else self.y

        preds = X_b @ w
        errors = preds - y_b
        return (2.0 / len(y_b)) * (X_b.T @ errors)

class LogLoss(Objective):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__()
        self.X = np.c_[np.ones(X.shape[0]), X]
        self.y = y.astype(np.float64)
        self.tags = {"classification", "logistic-regression", "convex"}

    def get_n_samples(self) -> int:
        return len(self.y)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_value(self, w: NDArray[np.float64], indices=None) -> float:
        X_b = self.X[indices] if indices is not None else self.X
        y_b = self.y[indices] if indices is not None else self.y

        preds = self._sigmoid(X_b @ w)
        preds = np.clip(preds, 1e-15, 1.0 - 1e-15)

        return float(-np.mean(y_b * np.log(preds) + (1.0 - y_b) * np.log(1.0 - preds)))

    def _compute_grad(self, w: NDArray[np.float64], indices=None) -> NDArray[np.float64]:
        X_b = self.X[indices] if indices is not None else self.X
        y_b = self.y[indices] if indices is not None else self.y
        
        preds = self._sigmoid(X_b @ w)
        return (X_b.T @ (preds - y_b)) / len(y_b)