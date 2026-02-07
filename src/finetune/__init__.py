"""
Fine-tuning module for LFM2-350M with Unsloth and LoRA.

Provides model setup, LoRA adapter configuration, and training utilities
for efficient fine-tuning on physics prediction tasks.
"""

from .lfm2_trainer import (
    setup_lfm2_model,
    add_lora_adapters,
    prepare_dataset,
    train_stage,
)

from .physics_loss import (
    PhysicsAuxiliaryLoss,
    compute_physics_loss,
    extract_physics_from_text,
)

__all__ = [
    # Model setup
    "setup_lfm2_model",
    "add_lora_adapters",
    # Data preparation
    "prepare_dataset",
    # Training
    "train_stage",
    # Physics loss
    "PhysicsAuxiliaryLoss",
    "compute_physics_loss",
    "extract_physics_from_text",
]
