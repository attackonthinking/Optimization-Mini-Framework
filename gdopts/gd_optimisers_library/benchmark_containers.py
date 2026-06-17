from dataclasses import dataclass

@dataclass
class RunMetrics:
    wall_time_sec: float
    peak_memory_bytes: int
    current_memory_bytes: int
    function_calls: int | None
    gradient_calls: int | None
    samples_seen: int | None


@dataclass
class GradientBenchmarkResult:
    method_name: str
    dimension: int
    epsilon: float | None
    grad_error_l2: float
    grad_error_inf: float
    run_metrics: RunMetrics
    
    
@dataclass
class OptimizerBenchmarkResult:
    optimizer_name: str
    batch_size: int | None
    step_strategy_name: str
    regularization_name: str | None
    hyperparameters: dict[str, float | int | str | bool]
    final_f_val: float
    best_f_val: float
    final_grad_norm: float
    iterations: int
    success: bool
    stop_reason: str
    run_metrics: RunMetrics

