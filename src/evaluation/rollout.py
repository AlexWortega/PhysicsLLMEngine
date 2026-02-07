"""
Rollout evaluator for autoregressive multi-step physics prediction.

Performs autoregressive evaluation by feeding model predictions back as input
for multi-step rollouts. Tracks per-step error accumulation and detects
divergence when errors exceed thresholds.

Key features:
- Autoregressive prediction: model output becomes next input
- Per-step error tracking for error accumulation curves
- Divergence detection when error exceeds threshold
- Memory-efficient for 100+ frame rollouts
"""

import re
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from src.evaluation.metrics import MetricsComputer


class RolloutEvaluator:
    """
    Autoregressive multi-step rollout evaluator.

    Performs autoregressive evaluation where model predictions are fed back
    as context for subsequent predictions. Tracks error accumulation and
    detects divergence points.
    """

    def __init__(
        self,
        metrics_computer: Optional[MetricsComputer] = None,
        divergence_threshold: Optional[float] = None,
    ):
        """
        Initialize RolloutEvaluator.

        Args:
            metrics_computer: MetricsComputer instance for computing metrics.
                             Creates one if not provided.
            divergence_threshold: Error threshold for divergence detection.
                                 Default: auto-computed as 10x mean single-step MSE.
        """
        self.metrics = metrics_computer or MetricsComputer()
        self.divergence_threshold = divergence_threshold
        self._auto_threshold = divergence_threshold is None

    def evaluate_rollout(
        self,
        model: Any,
        initial_context: str,
        ground_truth_frames: List[Dict[str, Any]],
        max_steps: int = 100,
        masses: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Perform autoregressive rollout evaluation.

        Generates predictions autoregressively by feeding each prediction
        back as context for the next step. Compares against ground truth
        at each step.

        Args:
            model: Model with generate method (GPT or fine-tuned LLM)
            initial_context: Scene header + initial frames as text
            ground_truth_frames: List of ground truth frame dicts to predict
            max_steps: Maximum number of rollout steps (default 100)
            masses: Optional object masses for conservation metrics.
                   If None, uses uniform mass of 1.0.

        Returns:
            Dictionary with:
                - step_errors: List of position MSE per step
                - velocity_errors: List of velocity MSE per step
                - cumulative_error: Sum of all step errors
                - divergence_step: First step where error exceeded threshold (or None)
                - predictions: List of predicted frame texts (for visualization)
                - energy_violations: List of energy violations per step
                - momentum_violations: List of momentum violations per step
        """
        num_steps = min(max_steps, len(ground_truth_frames))

        # Initialize results tracking
        step_errors: List[float] = []
        velocity_errors: List[float] = []
        energy_violations: List[float] = []
        momentum_violations: List[float] = []
        predictions: List[str] = []

        # Track context for autoregressive generation
        context = initial_context
        divergence_step: Optional[int] = None

        # Infer number of objects from first ground truth frame
        num_objects = len(ground_truth_frames[0].get("objects", []))
        if masses is None:
            masses = np.ones(num_objects)

        for step in range(num_steps):
            gt_frame = ground_truth_frames[step]

            # Generate prediction
            pred_text = self._predict_next_frame(model, context)
            predictions.append(pred_text)

            # Extract physics from prediction and ground truth
            pred_positions, pred_velocities = self._extract_physics(pred_text, num_objects)
            gt_positions, gt_velocities = self._extract_physics_from_frame(gt_frame)

            # Handle extraction failures
            if pred_positions is None or gt_positions is None:
                # Use large error for failed extractions
                step_errors.append(float("inf"))
                velocity_errors.append(float("inf"))
                energy_violations.append(0.0)
                momentum_violations.append(0.0)

                if divergence_step is None:
                    divergence_step = step

                # Update context with prediction anyway for continuation
                context = self._update_context(context, pred_text)
                continue

            # Compute metrics
            pos_mse = self.metrics.compute_position_mse(pred_positions, gt_positions)
            vel_mse = self.metrics.compute_velocity_mse(pred_velocities, gt_velocities)

            step_errors.append(pos_mse)
            velocity_errors.append(vel_mse)

            # Conservation metrics
            energy_result = self.metrics.compute_energy_conservation(
                masses, gt_velocities, pred_velocities
            )
            momentum_result = self.metrics.compute_momentum_conservation(
                masses, gt_velocities, pred_velocities
            )

            energy_violations.append(energy_result["energy_violation"])
            momentum_violations.append(momentum_result["momentum_violation"])

            # Check divergence
            threshold = self._get_divergence_threshold(step_errors)
            if divergence_step is None and pos_mse > threshold:
                divergence_step = step

            # Update context for next step (autoregressive)
            context = self._update_context(context, pred_text)

        # Compute summary statistics
        finite_errors = [e for e in step_errors if np.isfinite(e)]
        cumulative_error = sum(finite_errors) if finite_errors else float("inf")

        return {
            "step_errors": step_errors,
            "velocity_errors": velocity_errors,
            "cumulative_error": cumulative_error,
            "divergence_step": divergence_step,
            "predictions": predictions,
            "energy_violations": energy_violations,
            "momentum_violations": momentum_violations,
            "num_steps": num_steps,
        }

    def _get_divergence_threshold(self, step_errors: List[float]) -> float:
        """
        Get divergence threshold, computing auto-threshold if needed.

        Auto-threshold: 10x mean of first 5 steps (or all if fewer).

        Args:
            step_errors: List of step errors so far

        Returns:
            Divergence threshold value
        """
        if not self._auto_threshold:
            return self.divergence_threshold

        # Need at least some errors to compute auto-threshold
        finite_errors = [e for e in step_errors if np.isfinite(e)]
        if len(finite_errors) < 1:
            return float("inf")  # No threshold yet

        # Use first 5 steps as baseline
        baseline_errors = finite_errors[: min(5, len(finite_errors))]
        mean_error = np.mean(baseline_errors)

        return 10.0 * mean_error if mean_error > 0 else 1.0

    def _predict_next_frame(self, model: Any, context: str) -> str:
        """
        Call model to predict next frame.

        Handles both fine-tuned models (with tokenizer) and from-scratch GPT.

        Args:
            model: Model instance with generate method
            context: Current context string

        Returns:
            Predicted frame text
        """
        # Check if model has a callable tokenizer (HF-style model)
        if hasattr(model, "tokenizer") and callable(model.tokenizer):
            return self._predict_with_tokenizer(model, context)

        # Otherwise use generate method (from-scratch GPT with PhysicsTokenizer)
        return self._predict_with_generate(model, context)

    def _predict_with_tokenizer(self, model: Any, context: str) -> str:
        """Generate prediction using a model with tokenizer attribute."""
        import torch

        tokenizer = model.tokenizer

        # Tokenize input
        inputs = tokenizer(
            context + "\nPredict next frame:",
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        # Move to model device if needed
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.1,  # Low temperature for deterministic output
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the predicted frame
        if "Frame " in generated_text:
            # Find last frame prediction
            frame_start = generated_text.rfind("Frame ")
            return generated_text[frame_start:]

        return generated_text

    def _predict_with_generate(self, model: Any, context: str) -> str:
        """Generate prediction using model's generate method directly."""
        import torch

        # Check for tokenizer as separate attribute
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            # Fallback: try to use physics tokenizer
            try:
                from src.scratch.physics_tokenizer import PhysicsTokenizer

                tokenizer = PhysicsTokenizer()
            except ImportError:
                raise RuntimeError("No tokenizer available for model")

        # Tokenize
        tokens = tokenizer.encode(context + "\nPredict next frame:")
        idx = torch.tensor([tokens], dtype=torch.long)

        # Move to model device
        device = next(model.parameters()).device
        idx = idx.to(device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                idx,
                max_new_tokens=500,
                temperature=0.1,
            )

        # Decode
        output_tokens = output_ids[0].tolist()
        generated_text = tokenizer.decode(output_tokens)

        # Extract frame
        if "Frame " in generated_text:
            frame_start = generated_text.rfind("Frame ")
            return generated_text[frame_start:]

        return generated_text

    def _extract_physics(
        self, frame_text: str, num_objects: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract position and velocity arrays from frame text.

        Args:
            frame_text: Frame text in format:
                "obj_0: pos=(X, Y), vel=(VX, VY)"
            num_objects: Expected number of objects

        Returns:
            Tuple of (positions, velocities) arrays, each shape (num_objects, 2).
            Returns (None, None) if parsing fails.
        """
        # Pattern to match position and velocity
        obj_pattern = r"obj_(\d+):\s*pos=\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\),\s*vel=\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)"

        matches = re.findall(obj_pattern, frame_text)

        if len(matches) < num_objects:
            return None, None

        try:
            positions = np.zeros((num_objects, 2))
            velocities = np.zeros((num_objects, 2))

            for match in matches[:num_objects]:
                obj_id = int(match[0])
                if obj_id < num_objects:
                    positions[obj_id] = [float(match[1]), float(match[2])]
                    velocities[obj_id] = [float(match[3]), float(match[4])]

            return positions, velocities
        except (ValueError, IndexError):
            return None, None

    def _extract_physics_from_frame(
        self, frame_dict: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract physics from a frame dictionary (ground truth format).

        Args:
            frame_dict: Frame dict with "objects" list containing position/velocity

        Returns:
            Tuple of (positions, velocities) arrays
        """
        objects = frame_dict.get("objects", [])
        if not objects:
            return None, None

        try:
            num_objects = len(objects)
            positions = np.zeros((num_objects, 2))
            velocities = np.zeros((num_objects, 2))

            for obj in objects:
                obj_id = obj.get("id", 0)
                pos = obj.get("position", {})
                vel = obj.get("velocity", {})

                positions[obj_id] = [pos.get("x", 0), pos.get("y", 0)]
                velocities[obj_id] = [vel.get("x", 0), vel.get("y", 0)]

            return positions, velocities
        except (ValueError, IndexError, KeyError):
            return None, None

    def _update_context(self, context: str, predicted_frame: str) -> str:
        """
        Append predicted frame to context for next step.

        Manages context window to prevent memory issues on long rollouts.
        Keeps header and last N frames to stay within reasonable limits.

        Args:
            context: Current context string
            predicted_frame: Predicted frame to append

        Returns:
            Updated context string
        """
        # Simple append for now - context window management happens
        # at generation time via truncation
        updated = context.strip() + "\n\n" + predicted_frame.strip()

        # Rough check for context length - keep last ~50KB
        max_context_chars = 50000
        if len(updated) > max_context_chars:
            # Keep header (first part up to first Frame) + recent frames
            if "Frame 0:" in updated:
                header_end = updated.find("Frame 0:")
                header = updated[:header_end]
                # Keep last portion
                remaining_space = max_context_chars - len(header) - 100
                frames_portion = updated[-remaining_space:]
                # Find clean frame boundary
                frame_start = frames_portion.find("Frame ")
                if frame_start > 0:
                    frames_portion = frames_portion[frame_start:]
                updated = header + frames_portion

        return updated

    def batch_evaluate_rollouts(
        self,
        model: Any,
        scenes: List[Tuple[str, List[Dict[str, Any]]]],
        max_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Evaluate multiple scenes and aggregate results.

        Args:
            model: Model to evaluate
            scenes: List of (initial_context, ground_truth_frames) tuples
            max_steps: Maximum steps per scene

        Returns:
            Aggregated rollout statistics across all scenes
        """
        all_step_errors: List[List[float]] = []
        all_divergence_steps: List[Optional[int]] = []
        all_cumulative_errors: List[float] = []

        for context, gt_frames in scenes:
            result = self.evaluate_rollout(model, context, gt_frames, max_steps)
            all_step_errors.append(result["step_errors"])
            all_divergence_steps.append(result["divergence_step"])
            all_cumulative_errors.append(result["cumulative_error"])

        # Aggregate
        finite_cumulative = [e for e in all_cumulative_errors if np.isfinite(e)]
        diverged_count = sum(1 for d in all_divergence_steps if d is not None)

        # Mean step errors (aligned by step)
        max_len = max(len(e) for e in all_step_errors) if all_step_errors else 0
        mean_step_errors = []
        for step in range(max_len):
            step_vals = []
            for errors in all_step_errors:
                if step < len(errors) and np.isfinite(errors[step]):
                    step_vals.append(errors[step])
            if step_vals:
                mean_step_errors.append(float(np.mean(step_vals)))
            else:
                mean_step_errors.append(float("inf"))

        return {
            "mean_step_errors": mean_step_errors,
            "mean_cumulative_error": float(np.mean(finite_cumulative)) if finite_cumulative else float("inf"),
            "std_cumulative_error": float(np.std(finite_cumulative)) if finite_cumulative else 0.0,
            "diverged_scenes": diverged_count,
            "total_scenes": len(scenes),
            "divergence_rate": diverged_count / len(scenes) if scenes else 0.0,
        }
