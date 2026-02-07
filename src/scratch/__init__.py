"""From-scratch GPT training module for physics prediction.

This module provides infrastructure for training a GPT model from scratch
on physics simulation data, as a comparison to fine-tuning approaches.

Key components:
- PhysicsTokenizer: Wraps Phase 1 digit-level tokenizer for nanochat compatibility
- PhysicsTask: Dataset interface for nanochat training
- GPTConfig: Model architecture configuration with muP scaling support
"""

from src.scratch.physics_tokenizer import PhysicsTokenizer, get_physics_vocab
from src.scratch.nanochat_adapter import PhysicsTask, setup_nanochat_model

__all__ = [
    "PhysicsTokenizer",
    "get_physics_vocab",
    "PhysicsTask",
    "setup_nanochat_model",
]
