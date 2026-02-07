"""
Training infrastructure module for physics LLM.

Provides data loading, curriculum learning, checkpointing, and W&B integration.
"""

from .data_loader import (
    load_physics_scene,
    format_training_example,
    jsonl_to_training_examples,
)

from .curriculum import (
    CurriculumDataset,
    get_curriculum_stages,
)

from .checkpointing import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
)

from .wandb_utils import (
    setup_wandb,
    log_training_step,
    log_validation,
    save_checkpoint_artifact,
    log_curriculum_stage,
    finish_run,
)

__all__ = [
    # Data loading
    "load_physics_scene",
    "format_training_example",
    "jsonl_to_training_examples",
    # Curriculum
    "CurriculumDataset",
    "get_curriculum_stages",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    # W&B
    "setup_wandb",
    "log_training_step",
    "log_validation",
    "save_checkpoint_artifact",
    "log_curriculum_stage",
    "finish_run",
]
