from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np


class Objective(ABC):
    def __init__(self):
        self.function_calls = 0
        self.gradient_calls = 0
        self.samples_seen = 0
        self.tags = set()

    def __call__(self, x: NDArray[np.float64], indices=None) -> float:
        self.function_calls += 1
        self.samples_seen += self._get_batch_size(indices)
        return self._compute_value(x, indices)

    def grad(self, x: NDArray[np.float64], indices=None) -> NDArray[np.float64]:
        self.gradient_calls += 1
        self.samples_seen += self._get_batch_size(indices)
        return self._compute_grad(x, indices)

    def _get_batch_size(self, indices=None) -> int:
        if indices is None:
            return 1
        return len(indices)

    @abstractmethod
    def _compute_value(self, x: NDArray[np.float64], indices=None) -> float:
        pass

    @abstractmethod
    def _compute_grad(self, x: NDArray[np.float64], indices=None) -> NDArray[np.float64]:
        pass

    def reset_counters(self):
        self.function_calls = 0
        self.gradient_calls = 0
        self.samples_seen = 0