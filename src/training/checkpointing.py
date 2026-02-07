"""
Checkpointing module for training state save/resume.

Saves complete training state including model, optimizer, scheduler, and RNG states
for reproducible resume.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save complete training checkpoint for resume.

    Saves:
    - Model state dict
    - Optimizer state dict
    - Scheduler state dict (if provided)
    - Training progress (epoch, step, loss)
    - RNG states for reproducibility
    - Config (if provided)

    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch
        step: Current global step
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoint
        config: Training config dict (optional)

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Collect RNG states
    rng_state = {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    # Try to get numpy and random states if available
    try:
        import numpy as np
        rng_state["numpy"] = np.random.get_state()
    except ImportError:
        pass

    try:
        import random
        rng_state["python"] = random.getstate()
    except ImportError:
        pass

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "rng_state": rng_state,
        "config": config,
    }

    # Save numbered checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Also save as latest for easy resume
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Tuple[int, int, float]:
    """
    Load checkpoint and restore all training state.

    Restores:
    - Model state dict
    - Optimizer state dict
    - Scheduler state dict (if scheduler provided)
    - RNG states for reproducibility

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler (optional)
        device: Device to load tensors to

    Returns:
        Tuple of (epoch, step, loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if available
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNG states for reproducibility
    rng_state = checkpoint.get("rng_state", {})

    if rng_state.get("torch") is not None:
        torch.set_rng_state(rng_state["torch"])

    if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    if rng_state.get("numpy") is not None:
        try:
            import numpy as np
            np.random.set_state(rng_state["numpy"])
        except ImportError:
            pass

    if rng_state.get("python") is not None:
        try:
            import random
            random.setstate(rng_state["python"])
        except ImportError:
            pass

    return checkpoint["epoch"], checkpoint["step"], checkpoint["loss"]


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Looks for checkpoint_latest.pt first, then finds highest epoch/step.

    Args:
        checkpoint_dir: Directory to search

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    # Check for latest.pt first
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    if latest_path.exists():
        return str(latest_path)

    # Find all checkpoint files
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch*_step*.pt"))

    if not checkpoints:
        return None

    # Parse epoch and step from filename, sort by (epoch, step)
    def parse_checkpoint(path: Path) -> Tuple[int, int]:
        name = path.stem  # checkpoint_epoch{N}_step{M}
        parts = name.split("_")
        epoch = int(parts[1].replace("epoch", ""))
        step = int(parts[2].replace("step", ""))
        return (epoch, step)

    checkpoints.sort(key=parse_checkpoint, reverse=True)
    return str(checkpoints[0])


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading full state.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dict with checkpoint metadata (epoch, step, loss, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    return {
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
        "loss": checkpoint.get("loss"),
        "config": checkpoint.get("config"),
        "has_scheduler": checkpoint.get("scheduler_state_dict") is not None,
    }
