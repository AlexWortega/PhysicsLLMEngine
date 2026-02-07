"""GPT model configuration with muP scaling support.

Provides GPTConfig dataclass for model architecture and functions
for applying muP (Maximal Update Parametrization) scaling for
hyperparameter transfer across model sizes.
"""

from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn

# Try to import mup, but allow running without it
try:
    import mup
    from mup import MuReadout, MuAdamW, set_base_shapes

    MUP_AVAILABLE = True
except ImportError:
    MUP_AVAILABLE = False
    mup = None


@dataclass
class GPTConfig:
    """Configuration for GPT model architecture.

    Attributes:
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        vocab_size: Vocabulary size
        block_size: Maximum sequence length (context window)
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
    """

    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    vocab_size: int = 100  # Will be set from tokenizer
    block_size: int = 1024
    dropout: float = 0.1
    bias: bool = False


def create_model_config(size: str = "small") -> GPTConfig:
    """Create model configuration for specified size.

    Args:
        size: Model size - "small" (~10M params) or "medium" (~85M params)

    Returns:
        GPTConfig with appropriate hyperparameters

    Raises:
        ValueError: If size is not recognized
    """
    if size == "small":
        return GPTConfig(
            n_layer=6,
            n_head=6,
            n_embd=384,
            vocab_size=100,
            block_size=1024,
            dropout=0.1,
            bias=False,
        )
    elif size == "medium":
        return GPTConfig(
            n_layer=12,
            n_head=12,
            n_embd=768,
            vocab_size=100,
            block_size=1024,
            dropout=0.1,
            bias=False,
        )
    else:
        raise ValueError(f"Unknown model size: {size}. Choose 'small' or 'medium'.")


def apply_mup_scaling(
    model: nn.Module,
    base_model: nn.Module,
) -> nn.Module:
    """Apply muP scaling to model for hyperparameter transfer.

    muP (Maximal Update Parametrization) enables hyperparameters
    tuned on small models to transfer to larger models.

    Key changes:
    - Attention uses 1/d scaling instead of 1/sqrt(d)
    - Output layer has special scaling
    - Width-dependent LR scaling for embeddings

    Args:
        model: Target model to apply scaling to
        base_model: Base (smaller) model for shape comparison

    Returns:
        Model with muP scaling applied
    """
    if not MUP_AVAILABLE:
        print("Warning: mup not installed, skipping muP scaling")
        return model

    # Set base shapes from the smaller model
    set_base_shapes(model, base_model)

    return model


def create_mup_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """Create muP-compatible optimizer.

    Uses MuAdamW which applies proper LR scaling per parameter
    based on muP rules.

    Args:
        model: Model to optimize
        lr: Base learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters

    Returns:
        MuAdamW optimizer (or regular AdamW if mup not available)
    """
    if not MUP_AVAILABLE:
        print("Warning: mup not installed, using standard AdamW")
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )

    return MuAdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )


def estimate_model_params(config: GPTConfig) -> int:
    """Estimate number of parameters for a model configuration.

    Args:
        config: Model configuration

    Returns:
        Estimated parameter count
    """
    # Embedding: vocab_size * n_embd
    embed_params = config.vocab_size * config.n_embd

    # Position embedding: block_size * n_embd
    pos_params = config.block_size * config.n_embd

    # Per transformer layer:
    # - Attention: 4 * n_embd^2 (Q, K, V, output projections)
    # - MLP: 8 * n_embd^2 (2 layers with 4x expansion)
    # - LayerNorm: 2 * n_embd (scale and shift)
    layer_params = (4 + 8) * config.n_embd**2 + 2 * config.n_embd
    total_layer_params = config.n_layer * layer_params

    # Final LayerNorm: n_embd
    final_ln_params = config.n_embd

    # Output head (tied with embeddings usually, but count separately)
    output_params = config.n_embd * config.vocab_size

    total = embed_params + pos_params + total_layer_params + final_ln_params + output_params

    return total
