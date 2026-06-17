
from abc import ABC, abstractmethod
from .data_containers import OptimizationState, OptimizationHistory

class StoppingCriterion(ABC):
    @abstractmethod
    def check(self, state: OptimizationState, history: OptimizationHistory) -> bool:
        pass

    @abstractmethod
    def reason(self) -> str:
        pass


class MaxIterationsCriterion(StoppingCriterion):
    def __init__(self, max_iter: int = 10000):
        self.max_iter = max_iter

    def check(self, state: OptimizationState, history: OptimizationHistory) -> bool:
        return state.iteration >= self.max_iter

    def reason(self) -> str:
        return f"Reached max iterations ({self.max_iter})"


class GradientNormCriterion(StoppingCriterion):
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def check(self, state: OptimizationState, history: OptimizationHistory) -> bool:
        return state.grad_norm is not None and state.grad_norm <= self.tolerance

    def reason(self) -> str:
        return f"Gradient norm is below tolerance ({self.tolerance})"


class ArgumentChangeCriterion(StoppingCriterion):
    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = tolerance

    def check(self, state: OptimizationState, history: OptimizationHistory) -> bool:
        if not history.param_change_norms:
            return False
        return history.param_change_norms[-1] <= self.tolerance

    def reason(self) -> str:
        return f"Argument change is below tolerance ({self.tolerance})"


class FunctionChangeCriterion(StoppingCriterion):
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance

    def check(self, state: OptimizationState, history: OptimizationHistory) -> bool:
        if len(history.f_values) < 2:
            return False
        change = abs(history.f_values[-1] - history.f_values[-2])
        return change <= self.tolerance

    def reason(self) -> str:
        return f"Function value change is below tolerance ({self.tolerance})"
    