from abc import ABC, abstractmethod
import numpy as np


class UpdateRule(ABC):
    @abstractmethod
    def get_update(self, x: np.ndarray, grad: np.ndarray, state) -> np.ndarray:
        pass

    def reset(self):
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__


class SGDRule(UpdateRule):
    def get_update(self, x: np.ndarray, grad: np.ndarray, state) -> np.ndarray:
        return grad

    def __repr__(self) -> str:
        return "SGDRule()"


class MomentumRule(UpdateRule):
    def __init__(self, beta: float = 0.9):
        self.beta = float(beta)
        self._velocity: np.ndarray | None = None

    def reset(self):
        self._velocity = None

    def get_update(self, x: np.ndarray, grad: np.ndarray, state) -> np.ndarray:
        if self._velocity is None:
            self._velocity = np.zeros_like(grad)
        self._velocity = self.beta * self._velocity + grad
        return self._velocity.copy()

    def __repr__(self) -> str:
        return f"MomentumRule(beta={self.beta})"


# ─────────────────────────────────────────────────────────────────────────────
# Nesterov Accelerated Gradient
# ─────────────────────────────────────────────────────────────────────────────

class NesterovRule(UpdateRule):
    def __init__(self, beta: float = 0.9):
        self.beta = float(beta)
        self._velocity: np.ndarray | None = None

    def reset(self):
        self._velocity = None

    def get_update(self, x: np.ndarray, grad: np.ndarray, state) -> np.ndarray:
        if self._velocity is None:
            self._velocity = np.zeros_like(grad)

        v_prev = self._velocity.copy()
        self._velocity = self.beta * self._velocity + grad
        update = self._velocity + self.beta * (self._velocity - v_prev)
        return update

    def __repr__(self) -> str:
        return f"NesterovRule(beta={self.beta})"


class AdaGradRule(UpdateRule):
    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self._G: np.ndarray | None = None

    def reset(self):
        self._G = None

    def get_update(self, x: np.ndarray, grad: np.ndarray, state) -> np.ndarray:
        if self._G is None:
            self._G = np.zeros_like(grad)
        self._G += grad ** 2
        return grad / (np.sqrt(self._G) + self.eps)

    def __repr__(self) -> str:
        return f"AdaGradRule(eps={self.eps})"


class RMSPropRule(UpdateRule):
    def __init__(self, rho: float = 0.9, eps: float = 1e-8):
        self.rho = float(rho)
        self.eps = float(eps)
        self._v: np.ndarray | None = None

    def reset(self):
        self._v = None

    def get_update(self, x: np.ndarray, grad: np.ndarray, state) -> np.ndarray:
        if self._v is None:
            self._v = np.zeros_like(grad)
        self._v = self.rho * self._v + (1.0 - self.rho) * grad ** 2
        return grad / (np.sqrt(self._v) + self.eps)

    def __repr__(self) -> str:
        return f"RMSPropRule(rho={self.rho}, eps={self.eps})"


class AdamRule(UpdateRule):
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self._m: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._t: int = 0

    def reset(self):
        self._m = None
        self._v = None
        self._t = 0

    def get_update(self, x: np.ndarray, grad: np.ndarray, state) -> np.ndarray:
        if self._m is None:
            self._m = np.zeros_like(grad)
            self._v = np.zeros_like(grad)

        self._t += 1
        self._m = self.beta1 * self._m + (1.0 - self.beta1) * grad
        self._v = self.beta2 * self._v + (1.0 - self.beta2) * grad ** 2

        m_hat = self._m / (1.0 - self.beta1 ** self._t)
        v_hat = self._v / (1.0 - self.beta2 ** self._t)

        return m_hat / (np.sqrt(v_hat) + self.eps)

    def __repr__(self) -> str:
        return f"AdamRule(beta1={self.beta1}, beta2={self.beta2}, eps={self.eps})"
    