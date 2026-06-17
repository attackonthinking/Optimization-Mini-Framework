from numpy.typing import NDArray
import numpy as np

from .objective import Objective

class Ellipsoid(Objective):
    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k
        self.tags = {"unimodal", "convex"}
        if self.k > 10.0:
            self.tags.add("ill-conditioned")

        self.global_minimum_x = np.array([0.0, 0.0])

    def _compute_value(self, x: NDArray[np.float64], indices=None) -> float:
        return float(x[0] ** 2 + self.k * x[1] ** 2)

    def _compute_grad(self, x: NDArray[np.float64], indices=None) -> NDArray[np.float64]:
        return np.array([
            2.0 * x[0],
            2.0 * self.k * x[1]
        ])


class Beale(Objective):
    def __init__(self):
        super().__init__()
        self.tags = {"unimodal", "non-convex", "curved-landscape"}
        self.global_minimum_x = np.array([3.0, 0.5])

    def _compute_value(self, x: NDArray[np.float64], indices=None) -> float:
        x1, x2 = x[0], x[1]
        term1 = (1.5 - x1 + x1 * x2) ** 2
        term2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
        term3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
        return float(term1 + term2 + term3)

    def _compute_grad(self, x: NDArray[np.float64], indices=None) -> NDArray[np.float64]:
        x1, x2 = x[0], x[1]

        u1 = 1.5 - x1 + x1 * x2
        u2 = 2.25 - x1 + x1 * x2 ** 2
        u3 = 2.625 - x1 + x1 * x2 ** 3

        dx1 = (
            2.0 * u1 * (x2 - 1.0)
            + 2.0 * u2 * (x2 ** 2 - 1.0)
            + 2.0 * u3 * (x2 ** 3 - 1.0)
        )
        dx2 = (
            2.0 * u1 * x1
            + 4.0 * u2 * x1 * x2
            + 6.0 * u3 * x1 * x2 ** 2
        )

        return np.array([dx1, dx2])


class Himmelblau(Objective):
    def __init__(self):
        super().__init__()
        self.tags = {"multimodal", "non-convex", "multiple-global-minima"}
        self.global_minimum_x = np.array([3.0, 2.0])

    def _compute_value(self, x: NDArray[np.float64], indices=None) -> float:
        x1, x2 = x[0], x[1]
        return float((x1 ** 2 + x2 - 11.0) ** 2 + (x1 + x2 ** 2 - 7.0) ** 2)

    def _compute_grad(self, x: NDArray[np.float64], indices=None) -> NDArray[np.float64]:
        x1, x2 = x[0], x[1]

        u1 = x1 ** 2 + x2 - 11.0
        u2 = x1 + x2 ** 2 - 7.0

        dx1 = 4.0 * x1 * u1 + 2.0 * u2
        dx2 = 2.0 * u1 + 4.0 * x2 * u2

        return np.array([dx1, dx2])


class Rastrigin(Objective):
    def __init__(self, dim: int = 2, A: float = 10.0):
        super().__init__()
        self.dim = dim
        self.A = A
        self.tags = {"multimodal", "non-convex", "many-local-minima"}
        if self.dim > 2:
            self.tags.add("n-dimensional")

        self.global_minimum_x = np.zeros(self.dim)

    def _compute_value(self, x: NDArray[np.float64], indices=None) -> float:
        return float(
            self.A * self.dim + np.sum(x ** 2 - self.A * np.cos(2.0 * np.pi * x))
        )

    def _compute_grad(self, x: NDArray[np.float64], indices=None) -> NDArray[np.float64]:
        return 2.0 * x + 2.0 * np.pi * self.A * np.sin(2.0 * np.pi * x)


class Ackley(Objective):
    def __init__(
        self,
        dim: int = 2,
        a: float = 20.0,
        b: float = 0.2,
        c: float = 2.0 * np.pi,
    ):
        super().__init__()
        self.dim = dim
        self.a = a
        self.b = b
        self.c = c
        self.tags = {"multimodal", "non-convex", "plateau"}
        if self.dim > 2:
            self.tags.add("n-dimensional")

        self.global_minimum_x = np.zeros(self.dim)

    def _compute_value(self, x: NDArray[np.float64], indices=None) -> float:
        sq_mean = np.mean(x ** 2)
        cos_mean = np.mean(np.cos(self.c * x))

        term1 = -self.a * np.exp(-self.b * np.sqrt(sq_mean))
        term2 = -np.exp(cos_mean)

        return float(term1 + term2 + self.a + np.e)

    def _compute_grad(self, x: NDArray[np.float64], indices=None) -> NDArray[np.float64]:
        sq_mean = np.mean(x ** 2)
        cos_mean = np.mean(np.cos(self.c * x))

        if np.isclose(sq_mean, 0.0):
            return np.zeros_like(x)

        sqrt_sq_mean = np.sqrt(sq_mean)

        term1 = (
            self.a
            * self.b
            * np.exp(-self.b * sqrt_sq_mean)
            * x
            / (self.dim * sqrt_sq_mean)
        )

        term2 = (
            np.exp(cos_mean)
            * self.c
            * np.sin(self.c * x)
            / self.dim
        )

        return term1 + term2
    