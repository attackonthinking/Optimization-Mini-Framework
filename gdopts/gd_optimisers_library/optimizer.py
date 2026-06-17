import time
import numpy as np


class MasterOptimizer:
    def __init__(
        self,
        objective,
        gradient_estimator,
        update_rule,
        step_strategy,
        stopping_criteria,
        regularizer=None,
        store_trajectory_if_dim_leq: int = 2,
    ):
        self.objective = objective
        self.gradient_estimator = gradient_estimator
        self.update_rule = update_rule
        self.step_strategy = step_strategy
        self.stopping_criteria = stopping_criteria
        self.regularizer = regularizer
        self.store_trajectory_if_dim_leq = store_trajectory_if_dim_leq

    def optimize(self, x0: np.ndarray):
        from .data_containers import OptimizationState, OptimizationHistory, OptimizationResult
        from .regularization import NoRegularization

        self.objective.reset_counters()
        regularizer = self.regularizer if self.regularizer is not None else NoRegularization()

        x = np.asarray(x0, dtype=float).copy()
        f_val = self.objective(x)
        grad = self.gradient_estimator.estimate(self.objective, x)
        grad = regularizer.apply(x, grad)
        grad_norm = float(np.linalg.norm(grad))

        store_traj = x.size <= self.store_trajectory_if_dim_leq
        history = OptimizationHistory(store_trajectory=store_traj)

        best_x = x.copy()
        best_f = float(f_val)
        stop_reason = ""
        success = False

        start = time.perf_counter()
        iteration = 0

        while True:
            state = OptimizationState(
                iteration=iteration,
                x=x.copy(),
                f_val=float(f_val),
                grad=grad.copy(),
                grad_norm=grad_norm,
                step_size=None,
            )

            step_size = float(self.step_strategy.get_step(self.objective, state))
            state.step_size = step_size

            update = self.update_rule.get_update(x, grad, state)
            x_new = x - step_size * update
            param_change_norm = float(np.linalg.norm(x_new - x))

            history.iterations.append(iteration)
            history.f_values.append(float(f_val))
            history.grad_norms.append(float(grad_norm))
            history.step_sizes.append(step_size)
            history.param_change_norms.append(param_change_norm)
            if history.store_trajectory:
                history.trajectory.append(x.copy())

            for criterion in self.stopping_criteria:
                if criterion.check(state, history):
                    stop_reason = criterion.reason()
                    success = criterion.__class__.__name__ != "MaxIterationsCriterion"
                    wall_time_sec = time.perf_counter() - start
                    return OptimizationResult(
                        x_best=best_x.copy(),
                        f_best=float(best_f),
                        x_final=x.copy(),
                        f_final=float(f_val),
                        grad_norm_final=float(grad_norm),
                        iterations=iteration,
                        success=success,
                        stop_reason=stop_reason,
                        history=history,
                        function_calls=self.objective.function_calls,
                        gradient_calls=self.objective.gradient_calls,
                        samples_seen=self.objective.samples_seen,
                        wall_time_sec=wall_time_sec,
                    )

            x = x_new
            f_val = self.objective(x)
            grad = self.gradient_estimator.estimate(self.objective, x)
            grad = regularizer.apply(x, grad)
            grad_norm = float(np.linalg.norm(grad))

            if f_val < best_f:
                best_f = float(f_val)
                best_x = x.copy()

            iteration += 1
            