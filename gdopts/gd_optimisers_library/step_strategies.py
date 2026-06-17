
from abc import ABC, abstractmethod
from .data_containers import OptimizationState, OptimizationHistory


class StepStrategy(ABC):
    @abstractmethod
    def get_step(self, function, state: OptimizationState) -> float:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__


class FixedStep(StepStrategy):
    def __init__(self, step_size: float = 0.01):
        self.step_size = float(step_size)

    def get_step(self, function, state: OptimizationState) -> float:
        return self.step_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(step_size={self.step_size})"


class DecayingStep(StepStrategy):
    def __init__(self, initial_step: float = 0.1, decay_rate: float = 0.01):
        self.initial_step = float(initial_step)
        self.decay_rate = float(decay_rate)

    def get_step(self, function, state: OptimizationState) -> float:
        k = state.iteration
        return self.initial_step / (1.0 + self.decay_rate * k)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(initial_step={self.initial_step}, decay_rate={self.decay_rate})"
        )


class BacktrackingStep(StepStrategy):
    def __init__(
        self,
        initial_step: float = 1.0,
        shrink_factor: float = 0.5,
        armijo_constant: float = 1e-4,
        min_step: float = 1e-12,
        max_backtracking_steps: int = 50,
    ):
        self.initial_step = float(initial_step)
        self.shrink_factor = float(shrink_factor)
        self.armijo_constant = float(armijo_constant)
        self.min_step = float(min_step)
        self.max_backtracking_steps = int(max_backtracking_steps)

    def get_step(self, function, state: OptimizationState) -> float:
        alpha = self.initial_step

        x = state.x
        grad = state.grad
        f_x = state.f_val
        grad_sq_norm = state.grad_norm ** 2

        for _ in range(self.max_backtracking_steps):
            candidate_x = x - alpha * grad
            candidate_f = function(candidate_x)

            if candidate_f <= f_x - self.armijo_constant * alpha * grad_sq_norm:
                return alpha

            alpha *= self.shrink_factor

            if alpha < self.min_step:
                return self.min_step

        return max(alpha, self.min_step)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"initial_step={self.initial_step}, "
            f"shrink_factor={self.shrink_factor}, "
            f"armijo_constant={self.armijo_constant}, "
            f"min_step={self.min_step}, "
            f"max_backtracking_steps={self.max_backtracking_steps})"
        )
