"""
Weights & Biases integration utilities for experiment tracking.

Provides setup, logging, and artifact management for training runs.
"""

import os
from typing import Dict, Any, Optional, List

# Try to import wandb, but allow running without it
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def setup_wandb(
    config: Dict[str, Any],
    project: str = "physics-llm",
    tags: Optional[List[str]] = None,
    name: Optional[str] = None,
    mode: Optional[str] = None,
) -> Optional[Any]:
    """
    Initialize W&B run with config logging.

    Args:
        config: Training configuration dict
        project: W&B project name
        tags: Additional tags for the run
        name: Run name (optional, auto-generated if not provided)
        mode: W&B mode ("online", "offline", "disabled")

    Returns:
        wandb.Run object, or None if W&B not available/disabled
    """
    if not WANDB_AVAILABLE:
        print("Warning: wandb not installed, skipping W&B initialization")
        return None

    # Determine mode
    if mode is None:
        # Auto-detect from environment
        if os.environ.get("WANDB_MODE") == "disabled":
            mode = "disabled"
        elif os.environ.get("WANDB_API_KEY"):
            mode = "online"
        else:
            mode = "offline"

    if mode == "disabled":
        return None

    # Build tags
    run_tags = ["training", config.get("model_type", "unknown")]
    if tags:
        run_tags.extend(tags)

    try:
        run = wandb.init(
            project=project,
            config=config,
            tags=run_tags,
            name=name,
            mode=mode,
        )
        return run
    except Exception as e:
        print(f"Warning: Failed to initialize W&B: {e}")
        return None


def log_training_step(
    step: int,
    loss: float,
    lr: float,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Log training step metrics.

    Args:
        step: Global step number
        loss: Training loss
        lr: Learning rate
        metrics: Additional metrics dict (optional)
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    log_dict = {
        "train/loss": loss,
        "train/learning_rate": lr,
        "train/step": step,
    }

    if metrics:
        for key, value in metrics.items():
            log_dict[f"train/{key}"] = value

    wandb.log(log_dict, step=step)


def log_validation(
    step: int,
    val_loss: float,
    physics_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Log validation metrics including physics-specific ones.

    Args:
        step: Global step number
        val_loss: Validation loss
        physics_metrics: Dict with physics metrics (position_mse, velocity_mse, energy_error)
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    log_dict = {
        "val/loss": val_loss,
    }

    if physics_metrics:
        log_dict["val/position_mse"] = physics_metrics.get("position_mse", 0)
        log_dict["val/velocity_mse"] = physics_metrics.get("velocity_mse", 0)
        log_dict["val/energy_error"] = physics_metrics.get("energy_error", 0)

        # Add any other metrics with val/ prefix
        for key, value in physics_metrics.items():
            if key not in ("position_mse", "velocity_mse", "energy_error"):
                log_dict[f"val/{key}"] = value

    wandb.log(log_dict, step=step)


def save_checkpoint_artifact(
    checkpoint_path: str,
    epoch: int,
    step: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Save checkpoint as W&B artifact.

    Args:
        checkpoint_path: Path to checkpoint file
        epoch: Epoch number
        step: Step number
        metadata: Additional metadata for artifact

    Returns:
        Artifact name, or None if W&B not available
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return None

    artifact_name = f"checkpoint-epoch{epoch}-step{step}"

    try:
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata=metadata or {},
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        return artifact_name
    except Exception as e:
        print(f"Warning: Failed to save W&B artifact: {e}")
        return None


def log_curriculum_stage(
    stage_idx: int,
    num_scenes: int,
    complexity_range: tuple,
) -> None:
    """
    Log curriculum stage transition.

    Args:
        stage_idx: Stage index (0-based)
        num_scenes: Number of scenes in stage
        complexity_range: Tuple of (min_complexity, max_complexity)
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    complexity_min, complexity_max = complexity_range

    wandb.log({
        "curriculum/stage": stage_idx + 1,  # 1-indexed for display
        "curriculum/num_scenes": num_scenes,
        "curriculum/complexity_min": complexity_min,
        "curriculum/complexity_max": complexity_max,
    })


def finish_run() -> None:
    """Finish the current W&B run."""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
