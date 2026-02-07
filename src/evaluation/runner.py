"""
Evaluation runner that orchestrates all evaluation components.

Coordinates:
- Single-step metrics computation with curriculum breakdown
- Multi-step rollout evaluation
- Out-of-distribution (OOD) evaluation
- Trajectory visualization (GIF generation)
- Report generation
- W&B artifact logging

Provides a unified interface for complete model evaluation.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.evaluation.metrics import MetricsComputer
from src.evaluation.rollout import RolloutEvaluator
from src.evaluation.ood_generator import generate_ood_scene, OOD_TYPES
from src.evaluation.report import generate_evaluation_report

# Optional imports
try:
    from src.evaluation.visualization import (
        create_trajectory_gif, create_trajectory_overlay_gif, create_comparison_gif,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class EvaluationRunner:
    """
    Main orchestrator for complete model evaluation.

    Coordinates all evaluation components:
    - MetricsComputer for single-step metrics
    - RolloutEvaluator for multi-step prediction
    - OOD generator for out-of-distribution testing
    - Visualization for trajectory GIFs
    - Report generator for markdown reports
    - W&B for experiment tracking
    """

    def __init__(
        self,
        test_data_dir: str = "data/test",
        output_dir: str = "evaluation_results",
        use_wandb: bool = True,
    ):
        """
        Initialize EvaluationRunner.

        Args:
            test_data_dir: Directory containing test JSONL scenes
            output_dir: Directory for evaluation outputs
            use_wandb: Whether to log to W&B (default True)
        """
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.metrics = MetricsComputer()
        self.rollout = RolloutEvaluator(self.metrics)

        # W&B setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self._wandb_run = None

    def evaluate_model(
        self,
        model: Any,
        model_name: str,
        num_scenes: Optional[int] = None,
        rollout_steps: int = 100,
        num_gifs: int = 10,
        include_ood: bool = True,
        ood_scenes_per_type: int = 50,
    ) -> Dict[str, Any]:
        """
        Run complete evaluation for a single model.

        Steps:
        1. Single-step metrics on test set (with curriculum breakdown)
        2. Multi-step rollout evaluation
        3. Conservation metrics analysis
        4. OOD evaluation (if enabled)
        5. Generate representative GIFs
        6. Return aggregated results

        Args:
            model: Model to evaluate (must have generate method)
            model_name: Name for logging and reports
            num_scenes: Number of scenes to evaluate (None = all)
            rollout_steps: Maximum rollout steps (default 100)
            num_gifs: Number of trajectory GIFs to generate (default 10)
            include_ood: Whether to run OOD evaluation (default True)
            ood_scenes_per_type: Scenes per OOD type (default 50)

        Returns:
            Complete evaluation results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*60}")

        results = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
        }

        # 1. Single-step metrics
        print("\n[1/5] Computing single-step metrics...")
        single_step_results = self._evaluate_single_step(model, num_scenes)
        results.update(single_step_results)

        # 2. Multi-step rollout
        print("\n[2/5] Running multi-step rollouts...")
        rollout_results = self._evaluate_rollout(model, num_scenes, rollout_steps)
        results["rollout"] = rollout_results

        # 3. Conservation analysis is included in rollout results

        # 4. OOD evaluation
        if include_ood:
            print("\n[3/5] Running OOD evaluation...")
            ood_results = self._evaluate_ood(model, ood_scenes_per_type)
            results["ood"] = ood_results
        else:
            print("\n[3/5] Skipping OOD evaluation")
            results["ood"] = {}

        # 5. Generate visualizations
        gif_paths = []
        if VISUALIZATION_AVAILABLE and num_gifs > 0:
            print(f"\n[4/5] Generating {num_gifs} trajectory GIFs...")
            gif_paths = self._generate_visualizations(model, model_name, num_gifs, rollout_steps)
        else:
            print("\n[4/5] Skipping visualization (matplotlib not available or num_gifs=0)")

        results["visualization_paths"] = gif_paths

        # 6. Log to W&B
        if self.use_wandb:
            print("\n[5/5] Logging to W&B...")
            self._log_to_wandb(model_name, results, gif_paths)
        else:
            print("\n[5/5] Skipping W&B logging")

        print(f"\nEvaluation complete for {model_name}")
        return results

    def evaluate_all_models(
        self,
        models: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all models and generate comparison report.

        Args:
            models: Dictionary mapping model names to model instances
            **kwargs: Additional arguments passed to evaluate_model

        Returns:
            Dictionary mapping model names to their evaluation results
        """
        all_results = {}

        for model_name, model in models.items():
            results = self.evaluate_model(model, model_name, **kwargs)
            all_results[model_name] = results

        # Generate comparison report
        report_path = self.output_dir / "evaluation_report.md"
        visualization_paths = []
        for r in all_results.values():
            visualization_paths.extend(r.get("visualization_paths", []))

        generate_evaluation_report(
            all_results,
            str(report_path),
            test_set_size=self._count_test_scenes(),
            visualization_paths=visualization_paths,
        )

        print(f"\nComparison report generated: {report_path}")

        return all_results

    def _evaluate_single_step(
        self,
        model: Any,
        num_scenes: Optional[int],
    ) -> Dict[str, Any]:
        """
        Evaluate single-step prediction metrics.

        Computes position/velocity MSE and conservation metrics,
        aggregated by curriculum stage.
        """
        test_scenes = self._load_test_scenes(num_scenes)

        all_results = []
        object_counts = []

        for scene_path in test_scenes:
            try:
                header, frames = self._load_scene(scene_path)
                object_count = header.get("object_count", len(frames[0].get("objects", [])))

                # Build context from first frame(s)
                context = self._build_context(header, frames[:2])

                # Predict frame 2 (single step)
                if len(frames) > 2:
                    gt_frame = frames[2]
                    pred_text = self._generate_prediction(model, context)

                    # Extract physics and compute metrics
                    pred_pos, pred_vel = self.rollout._extract_physics(pred_text, object_count)
                    gt_pos, gt_vel = self.rollout._extract_physics_from_frame(gt_frame)

                    if pred_pos is not None and gt_pos is not None:
                        masses = np.ones(object_count)
                        metrics = self.metrics.compute_frame_metrics(
                            pred_pos, gt_pos, pred_vel, gt_vel, masses
                        )
                        all_results.append(metrics)
                        object_counts.append(object_count)

            except Exception as e:
                print(f"  Warning: Failed to evaluate {scene_path}: {e}")
                continue

        # Aggregate results
        if not all_results:
            return {
                "position_mse": {"mean": float("inf"), "std": 0.0},
                "velocity_mse": {"mean": float("inf"), "std": 0.0},
                "energy_violation": {"mean": 0.0, "std": 0.0},
                "momentum_violation": {"mean": 0.0, "std": 0.0},
            }

        # Overall aggregation
        aggregated = {}
        for metric in ["position_mse", "velocity_mse", "energy_violation", "momentum_violation"]:
            values = [r.get(metric, 0.0) for r in all_results]
            aggregated[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        # Curriculum breakdown
        by_curriculum = self.metrics.aggregate_all_metrics_by_curriculum(
            all_results, object_counts
        )
        aggregated["by_curriculum_stage"] = by_curriculum

        print(f"  Evaluated {len(all_results)} scenes")
        print(f"  Position MSE: {aggregated['position_mse']['mean']:.6f}")
        print(f"  Velocity MSE: {aggregated['velocity_mse']['mean']:.6f}")

        return aggregated

    def _evaluate_rollout(
        self,
        model: Any,
        num_scenes: Optional[int],
        rollout_steps: int,
    ) -> Dict[str, Any]:
        """
        Evaluate multi-step rollout performance.

        Runs autoregressive prediction for multiple steps and tracks
        error accumulation and divergence.
        """
        test_scenes = self._load_test_scenes(num_scenes)

        # Use subset for rollouts (expensive)
        rollout_scenes = test_scenes[: min(len(test_scenes), 100)]

        all_step_errors = []
        divergence_steps = []
        cumulative_errors = []

        for scene_path in rollout_scenes:
            try:
                header, frames = self._load_scene(scene_path)

                if len(frames) < rollout_steps + 1:
                    continue

                # Build context from first frame
                context = self._build_context(header, frames[:1])
                gt_frames = frames[1 : rollout_steps + 1]

                result = self.rollout.evaluate_rollout(
                    model, context, gt_frames, max_steps=rollout_steps
                )

                all_step_errors.append(result["step_errors"])
                divergence_steps.append(result["divergence_step"])
                cumulative_errors.append(result["cumulative_error"])

            except Exception as e:
                print(f"  Warning: Rollout failed for {scene_path}: {e}")
                continue

        # Aggregate
        if not all_step_errors:
            return {"step_errors": [], "divergence_step": None, "mean_cumulative_error": float("inf")}

        # Mean step errors
        max_steps = max(len(e) for e in all_step_errors)
        mean_errors = []
        for step in range(max_steps):
            step_vals = [
                e[step] for e in all_step_errors
                if step < len(e) and np.isfinite(e[step])
            ]
            mean_errors.append(float(np.mean(step_vals)) if step_vals else float("inf"))

        # Divergence stats
        finite_divergences = [d for d in divergence_steps if d is not None]
        diverged_count = len(finite_divergences)
        mean_divergence = float(np.mean(finite_divergences)) if finite_divergences else None

        finite_cumulative = [c for c in cumulative_errors if np.isfinite(c)]

        print(f"  Evaluated {len(all_step_errors)} rollouts")
        print(f"  Divergence rate: {diverged_count}/{len(all_step_errors)}")
        if mean_divergence is not None:
            print(f"  Mean divergence step: {mean_divergence:.1f}")

        return {
            "step_errors": mean_errors,
            "divergence_step": mean_divergence,
            "divergence_rate": diverged_count / len(all_step_errors) if all_step_errors else 0.0,
            "mean_cumulative_error": float(np.mean(finite_cumulative)) if finite_cumulative else float("inf"),
            "std_cumulative_error": float(np.std(finite_cumulative)) if finite_cumulative else 0.0,
        }

    def _evaluate_ood(
        self,
        model: Any,
        scenes_per_type: int,
    ) -> Dict[str, Any]:
        """
        Evaluate out-of-distribution generalization.

        Tests model on scenes outside the training distribution.
        """
        results_by_type = {}
        results_by_distance = []

        for ood_type in OOD_TYPES:
            type_results = []
            type_distances = []

            for i in range(scenes_per_type):
                try:
                    seed = 200000 + OOD_TYPES.index(ood_type) * 1000 + i
                    scene = generate_ood_scene(seed, ood_type, ood_level=1.5, num_frames=10)

                    header = scene["header"]
                    frames = scene["frames"]
                    ood_distance = scene["ood_distance"]

                    # Build context and evaluate
                    context = self._build_context_from_ood(header, frames[:2])
                    if len(frames) > 2:
                        gt_frame = frames[2]
                        pred_text = self._generate_prediction(model, context)

                        num_objects = header.get("num_objects", len(frames[0]["objects"]))
                        pred_pos, pred_vel = self.rollout._extract_physics(pred_text, num_objects)
                        gt_pos = np.array([[obj["position"]["x"], obj["position"]["y"]]
                                          for obj in gt_frame["objects"]])
                        gt_vel = np.array([[obj["velocity"]["x"], obj["velocity"]["y"]]
                                          for obj in gt_frame["objects"]])

                        if pred_pos is not None:
                            pos_mse = self.metrics.compute_position_mse(pred_pos, gt_pos)
                            type_results.append(pos_mse)
                            type_distances.append((ood_distance, pos_mse))

                except Exception as e:
                    continue

            # Aggregate for this OOD type
            if type_results:
                results_by_type[ood_type] = {
                    "position_mse": {
                        "mean": float(np.mean(type_results)),
                        "std": float(np.std(type_results)),
                    }
                }

            results_by_distance.extend(type_distances)

        # Bin by distance for graceful degradation analysis
        distance_bins = []
        if results_by_distance:
            distances = np.array([d[0] for d in results_by_distance])
            errors = np.array([d[1] for d in results_by_distance])

            # Create bins
            bins = np.linspace(0, 2, 5)
            for i in range(len(bins) - 1):
                mask = (distances >= bins[i]) & (distances < bins[i + 1])
                if mask.any():
                    distance_bins.append((
                        float((bins[i] + bins[i + 1]) / 2),
                        float(np.mean(errors[mask])),
                        float(np.std(errors[mask])),
                    ))

        print(f"  Evaluated {len(OOD_TYPES)} OOD types")
        for ood_type, metrics in results_by_type.items():
            print(f"  {ood_type}: {metrics['position_mse']['mean']:.6f}")

        return {
            "by_type": results_by_type,
            "by_distance": distance_bins,
        }

    def _generate_visualizations(
        self,
        model: Any,
        model_name: str,
        num_gifs: int,
        rollout_steps: int,
    ) -> List[str]:
        """Generate trajectory visualization GIFs."""
        if not VISUALIZATION_AVAILABLE:
            return []

        test_scenes = self._load_test_scenes(num_gifs)
        gif_paths = []

        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        for i, scene_path in enumerate(test_scenes[:num_gifs]):
            try:
                header, frames = self._load_scene(scene_path)

                if len(frames) < 20:
                    continue

                # Build context and run rollout
                context = self._build_context(header, frames[:1])
                gt_frames = frames[1:min(31, len(frames))]  # 30 frames max for GIF

                result = self.rollout.evaluate_rollout(model, context, gt_frames, max_steps=30)

                # Extract positions for visualization
                num_objects = len(gt_frames[0].get("objects", []))
                num_frames = min(len(result["predictions"]), len(gt_frames))

                pred_positions = np.zeros((num_frames, num_objects, 2))
                true_positions = np.zeros((num_frames, num_objects, 2))

                for j in range(num_frames):
                    pred_pos, _ = self.rollout._extract_physics(result["predictions"][j], num_objects)
                    true_pos, _ = self.rollout._extract_physics_from_frame(gt_frames[j])

                    if pred_pos is not None:
                        pred_positions[j] = pred_pos
                    if true_pos is not None:
                        true_positions[j] = true_pos

                # Generate side-by-side GIF
                gif_path = vis_dir / f"{model_name}_scene_{i:03d}.gif"
                create_trajectory_gif(
                    pred_positions,
                    true_positions,
                    str(gif_path),
                    title=f"{model_name} Scene {i}",
                )
                gif_paths.append(str(gif_path))

                # Generate comparison overlay GIF with error lines
                comp_path = vis_dir / f"{model_name}_scene_{i:03d}_comparison.gif"
                create_comparison_gif(
                    pred_positions,
                    true_positions,
                    str(comp_path),
                    step_errors=result["step_errors"][:num_frames],
                    title=f"{model_name} Scene {i}",
                )
                gif_paths.append(str(comp_path))
                print(f"  Generated: {gif_path} + comparison")

            except Exception as e:
                print(f"  Warning: Failed to generate GIF for scene {i}: {e}")
                continue

        return gif_paths

    def _log_to_wandb(
        self,
        model_name: str,
        results: Dict[str, Any],
        gif_paths: List[str],
    ):
        """Log results and artifacts to W&B."""
        if not self.use_wandb or not WANDB_AVAILABLE:
            return

        try:
            # Initialize run if not already
            if self._wandb_run is None:
                self._wandb_run = wandb.init(
                    project="physics-prediction-eval",
                    name=f"eval-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config={"model_name": model_name},
                )

            # Log metrics
            metrics_to_log = {
                f"{model_name}/position_mse": results.get("position_mse", {}).get("mean", 0),
                f"{model_name}/velocity_mse": results.get("velocity_mse", {}).get("mean", 0),
                f"{model_name}/energy_violation": results.get("energy_violation", {}).get("mean", 0),
                f"{model_name}/momentum_violation": results.get("momentum_violation", {}).get("mean", 0),
            }

            rollout = results.get("rollout", {})
            if rollout:
                metrics_to_log[f"{model_name}/rollout_cumulative_error"] = rollout.get("mean_cumulative_error", 0)
                metrics_to_log[f"{model_name}/divergence_rate"] = rollout.get("divergence_rate", 0)

            wandb.log(metrics_to_log)

            # Log GIFs as media
            for gif_path in gif_paths:
                if os.path.exists(gif_path):
                    wandb.log({f"{model_name}/trajectory": wandb.Video(gif_path)})

            # Create artifact with results
            artifact = wandb.Artifact(
                f"eval-{model_name}",
                type="evaluation",
                description=f"Evaluation results for {model_name}",
            )

            # Save results as JSON
            results_path = self.output_dir / f"{model_name}_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            artifact.add_file(str(results_path))

            # Add GIFs to artifact
            for gif_path in gif_paths:
                if os.path.exists(gif_path):
                    artifact.add_file(gif_path)

            wandb.log_artifact(artifact)
            print(f"  Logged to W&B: {len(gif_paths)} GIFs, {len(metrics_to_log)} metrics")

        except Exception as e:
            print(f"  Warning: W&B logging failed: {e}")

    def _load_test_scenes(self, num_scenes: Optional[int] = None) -> List[Path]:
        """Load test scene file paths."""
        if not self.test_data_dir.exists():
            print(f"Warning: Test data directory not found: {self.test_data_dir}")
            return []

        scenes = list(self.test_data_dir.glob("**/*.jsonl"))
        if num_scenes:
            scenes = scenes[:num_scenes]
        return scenes

    def _count_test_scenes(self) -> int:
        """Count total test scenes."""
        if not self.test_data_dir.exists():
            return 0
        return len(list(self.test_data_dir.glob("**/*.jsonl")))

    def _load_scene(self, scene_path: Path) -> Tuple[Dict, List[Dict]]:
        """Load a scene from JSONL file."""
        from src.training.data_loader import load_physics_scene
        return load_physics_scene(str(scene_path))

    def _build_context(self, header: Dict, frames: List[Dict]) -> str:
        """Build context string from header and frames."""
        lines = []

        # Scene description
        lines.append(f"Scene: {header.get('description', 'Physics scene')}")
        gravity = header.get("gravity", {"x": 0, "y": -981})
        lines.append(f"Gravity: ({gravity.get('x', 0)}, {gravity.get('y', -981)})")
        lines.append(f"Timestep: {header.get('timestep', 1/60):.5f}")
        lines.append("")

        # Frames
        for frame in frames:
            lines.append(f"Frame {frame['frame']}: {frame.get('description', '')}")
            for obj in frame.get("objects", []):
                pos = obj.get("position", {})
                vel = obj.get("velocity", {})
                lines.append(
                    f"  obj_{obj['id']}: pos=({pos.get('x', 0):.4f}, {pos.get('y', 0):.4f}), "
                    f"vel=({vel.get('x', 0):.4f}, {vel.get('y', 0):.4f})"
                )
            lines.append("")

        lines.append("Predict next frame:")

        return "\n".join(lines)

    def _build_context_from_ood(self, header: Dict, frames: List[Dict]) -> str:
        """Build context string from OOD scene (slightly different format)."""
        lines = []

        lines.append(f"Scene: OOD Test Scene (seed={header.get('seed', 0)})")
        gravity = header.get("gravity", {"x": 0, "y": -981})
        lines.append(f"Gravity: ({gravity.get('x', 0)}, {gravity.get('y', -981)})")
        lines.append(f"Timestep: {header.get('timestep', 1/60):.5f}")
        lines.append("")

        for idx, frame in enumerate(frames):
            lines.append(f"Frame {idx}:")
            for obj in frame.get("objects", []):
                pos = obj.get("position", {})
                vel = obj.get("velocity", {})
                obj_id = obj.get("id", 0)
                lines.append(
                    f"  obj_{obj_id}: pos=({pos.get('x', 0):.4f}, {pos.get('y', 0):.4f}), "
                    f"vel=({vel.get('x', 0):.4f}, {vel.get('y', 0):.4f})"
                )
            lines.append("")

        lines.append("Predict next frame:")

        return "\n".join(lines)

    def _generate_prediction(self, model: Any, context: str) -> str:
        """Generate prediction from model."""
        return self.rollout._predict_next_frame(model, context)

    def close(self):
        """Clean up resources."""
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
