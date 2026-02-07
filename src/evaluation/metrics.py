"""
Metrics computation for physics prediction evaluation.

Computes position/velocity MSE, energy/momentum conservation metrics,
and provides curriculum-based aggregation for analysis.

All computations use numpy arrays (not torch) for simplicity.
"""

from typing import Dict, List, Tuple
import numpy as np


# Curriculum stages from training (object count ranges)
CURRICULUM_STAGES: List[Tuple[int, int]] = [
    (10, 18),
    (18, 26),
    (26, 34),
    (34, 42),
    (42, 50),
]


class MetricsComputer:
    """
    Compute physics prediction metrics.

    Provides methods for:
    - Position and velocity MSE computation
    - Energy and momentum conservation analysis
    - Curriculum-based metric aggregation
    """

    def __init__(self):
        """Initialize MetricsComputer."""
        self.curriculum_stages = CURRICULUM_STAGES

    def compute_position_mse(
        self,
        pred: np.ndarray,
        true: np.ndarray,
    ) -> float:
        """
        Compute Mean Squared Error for positions.

        Args:
            pred: Predicted positions, shape (num_objects, 2)
            true: Ground truth positions, shape (num_objects, 2)

        Returns:
            Mean squared error as float
        """
        return float(np.mean((pred - true) ** 2))

    def compute_velocity_mse(
        self,
        pred: np.ndarray,
        true: np.ndarray,
    ) -> float:
        """
        Compute Mean Squared Error for velocities.

        Args:
            pred: Predicted velocities, shape (num_objects, 2)
            true: Ground truth velocities, shape (num_objects, 2)

        Returns:
            Mean squared error as float
        """
        return float(np.mean((pred - true) ** 2))

    def compute_energy_conservation(
        self,
        masses: np.ndarray,
        vel_t0: np.ndarray,
        vel_t1: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute energy conservation metrics.

        Computes kinetic energy at two time points and measures conservation.
        Energy violation only penalizes energy GAIN (entropy should not decrease).

        Args:
            masses: Object masses, shape (num_objects,)
            vel_t0: Velocities at initial time, shape (num_objects, 2)
            vel_t1: Velocities at final time, shape (num_objects, 2)

        Returns:
            Dictionary with:
                - initial_ke: Kinetic energy at t0
                - final_ke: Kinetic energy at t1
                - energy_change: final_ke - initial_ke (positive = gain)
                - energy_violation: max(0, energy_change) -- only penalize gain
        """
        # KE = 0.5 * m * |v|^2
        v0_squared = np.sum(vel_t0 ** 2, axis=-1)  # (num_objects,)
        v1_squared = np.sum(vel_t1 ** 2, axis=-1)  # (num_objects,)

        initial_ke = float(0.5 * np.sum(masses * v0_squared))
        final_ke = float(0.5 * np.sum(masses * v1_squared))

        energy_change = final_ke - initial_ke
        # Only penalize energy GAIN (positive change)
        energy_violation = max(0.0, energy_change)

        return {
            "initial_ke": initial_ke,
            "final_ke": final_ke,
            "energy_change": energy_change,
            "energy_violation": energy_violation,
        }

    def compute_momentum_conservation(
        self,
        masses: np.ndarray,
        vel_t0: np.ndarray,
        vel_t1: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute momentum conservation metrics.

        Computes total momentum at two time points and measures violation.

        Args:
            masses: Object masses, shape (num_objects,)
            vel_t0: Velocities at initial time, shape (num_objects, 2)
            vel_t1: Velocities at final time, shape (num_objects, 2)

        Returns:
            Dictionary with:
                - initial_momentum: Total momentum vector at t0, shape (2,)
                - final_momentum: Total momentum vector at t1, shape (2,)
                - momentum_violation: L2 norm of momentum difference
        """
        # p = m * v for each object, then sum
        # Expand masses for broadcasting: (num_objects,) -> (num_objects, 1)
        masses_expanded = masses[:, np.newaxis]

        initial_momentum = np.sum(masses_expanded * vel_t0, axis=0)  # (2,)
        final_momentum = np.sum(masses_expanded * vel_t1, axis=0)  # (2,)

        # L2 norm of difference
        momentum_diff = final_momentum - initial_momentum
        momentum_violation = float(np.sqrt(np.sum(momentum_diff ** 2)))

        return {
            "initial_momentum": initial_momentum,
            "final_momentum": final_momentum,
            "momentum_violation": momentum_violation,
        }

    def compute_frame_metrics(
        self,
        pred_positions: np.ndarray,
        true_positions: np.ndarray,
        pred_velocities: np.ndarray,
        true_velocities: np.ndarray,
        masses: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single frame comparison.

        Args:
            pred_positions: Predicted positions, shape (num_objects, 2)
            true_positions: Ground truth positions, shape (num_objects, 2)
            pred_velocities: Predicted velocities, shape (num_objects, 2)
            true_velocities: Ground truth velocities, shape (num_objects, 2)
            masses: Object masses, shape (num_objects,)

        Returns:
            Dictionary with all computed metrics:
                - position_mse: Position MSE
                - velocity_mse: Velocity MSE
                - energy_violation: Energy conservation violation
                - momentum_violation: Momentum conservation violation
        """
        position_mse = self.compute_position_mse(pred_positions, true_positions)
        velocity_mse = self.compute_velocity_mse(pred_velocities, true_velocities)

        # For conservation, compare predicted velocities to ground truth
        # This measures whether predictions respect physics laws
        energy_metrics = self.compute_energy_conservation(
            masses, true_velocities, pred_velocities
        )
        momentum_metrics = self.compute_momentum_conservation(
            masses, true_velocities, pred_velocities
        )

        return {
            "position_mse": position_mse,
            "velocity_mse": velocity_mse,
            "energy_violation": energy_metrics["energy_violation"],
            "momentum_violation": momentum_metrics["momentum_violation"],
        }

    def _get_curriculum_stage(self, object_count: int) -> str:
        """
        Get curriculum stage label for an object count.

        Args:
            object_count: Number of objects in scene

        Returns:
            Stage label like "10-18" or "unknown" if out of range
        """
        for low, high in self.curriculum_stages:
            if low <= object_count < high:
                return f"{low}-{high}"
        # Handle edge case: exactly 50 objects
        if object_count == 50:
            return "42-50"
        return "unknown"

    def aggregate_by_curriculum(
        self,
        results: List[Dict[str, float]],
        object_counts: List[int],
        metric_key: str = "position_mse",
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metric results by curriculum stage.

        Groups results by object count ranges and computes mean/std.

        Args:
            results: List of metric dictionaries (one per scene/sample)
            object_counts: List of object counts (same length as results)
            metric_key: Which metric to aggregate (default: position_mse)

        Returns:
            Dictionary mapping stage labels to aggregated stats:
            {
                "10-18": {"mean": float, "std": float, "count": int},
                "18-26": {"mean": float, "std": float, "count": int},
                ...
            }
        """
        if len(results) != len(object_counts):
            raise ValueError(
                f"Length mismatch: {len(results)} results vs {len(object_counts)} counts"
            )

        # Group values by stage
        stage_values: Dict[str, List[float]] = {}
        for result, count in zip(results, object_counts):
            stage = self._get_curriculum_stage(count)
            if stage not in stage_values:
                stage_values[stage] = []
            stage_values[stage].append(result.get(metric_key, 0.0))

        # Compute aggregates
        aggregated: Dict[str, Dict[str, float]] = {}
        for stage, values in stage_values.items():
            if values:
                arr = np.array(values)
                aggregated[stage] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "count": len(values),
                }

        return aggregated

    def aggregate_all_metrics_by_curriculum(
        self,
        results: List[Dict[str, float]],
        object_counts: List[int],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Aggregate all metrics by curriculum stage.

        Args:
            results: List of metric dictionaries
            object_counts: List of object counts

        Returns:
            Nested dictionary: {metric_name: {stage: {mean, std, count}}}
        """
        metric_keys = ["position_mse", "velocity_mse", "energy_violation", "momentum_violation"]

        all_aggregated = {}
        for key in metric_keys:
            all_aggregated[key] = self.aggregate_by_curriculum(
                results, object_counts, metric_key=key
            )

        return all_aggregated
