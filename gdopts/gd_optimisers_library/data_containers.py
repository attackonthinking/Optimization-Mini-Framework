import numpy as np
from dataclasses import dataclass, field


@dataclass
class OptimizationState:
    iteration: int
    x: np.ndarray
    f_val: float
    grad: np.ndarray | None
    grad_norm: float | None
    step_size: float | None = None


@dataclass
class OptimizationHistory:
    iterations: list[int] = field(default_factory=list)
    f_values: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    step_sizes: list[float] = field(default_factory=list)
    param_change_norms: list[float] = field(default_factory=list)
    trajectory: list[np.ndarray] = field(default_factory=list)
    store_trajectory: bool = False


@dataclass
class OptimizationResult:
    iterations: int
    success: bool
    stop_reason: str
    function_calls: int
    gradient_calls: int
    samples_seen: int
    x_best: np.ndarray
    f_best: float
    x_final: np.ndarray
    f_final: float
    grad_norm_final: float
    history: OptimizationHistory
    wall_time_sec: float | None = None