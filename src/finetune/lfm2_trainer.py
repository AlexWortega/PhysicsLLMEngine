"""
LFM2-350M model setup and training utilities using Unsloth.

Provides efficient fine-tuning with LoRA adapters for physics prediction.
Uses Unsloth for 2x speedup and reduced VRAM usage.
"""

from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

# Lazy imports to allow module to load without GPU dependencies
_UNSLOTH_AVAILABLE = None
_TRL_AVAILABLE = None


def _check_unsloth():
    """Check if unsloth is available."""
    global _UNSLOTH_AVAILABLE
    if _UNSLOTH_AVAILABLE is None:
        try:
            from unsloth import FastLanguageModel
            _UNSLOTH_AVAILABLE = True
        except ImportError:
            _UNSLOTH_AVAILABLE = False
    return _UNSLOTH_AVAILABLE


def _check_trl():
    """Check if trl is available."""
    global _TRL_AVAILABLE
    if _TRL_AVAILABLE is None:
        try:
            from trl import SFTTrainer, SFTConfig
            _TRL_AVAILABLE = True
        except ImportError:
            _TRL_AVAILABLE = False
    return _TRL_AVAILABLE


def setup_lfm2_model(
    max_seq_length: int = 4096,
    dtype: str = "bfloat16",
    load_in_4bit: bool = False,
) -> Tuple[Any, Any]:
    """
    Setup LFM2-350M model with Unsloth for efficient fine-tuning.

    Uses load_in_4bit=False by default for physics accuracy (research recommendation).
    A6000 has 48GB VRAM so memory is not a constraint.

    Args:
        max_seq_length: Maximum sequence length (default 4096)
        dtype: Data type for model weights ("bfloat16" or "float16")
        load_in_4bit: Whether to load in 4-bit quantization (False for accuracy)

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ImportError: If unsloth is not installed
    """
    if not _check_unsloth():
        raise ImportError(
            "unsloth is not installed. Install with: "
            "pip install git+https://github.com/unslothai/unsloth.git"
        )

    from unsloth import FastLanguageModel
    import torch

    # Map dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="LiquidAI/LFM2-350M",
        max_seq_length=max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
    )

    return model, tokenizer


def add_lora_adapters(
    model: Any,
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> Any:
    """
    Add LoRA adapters to model for efficient fine-tuning.

    Uses r=32 and alpha=64 per research recommendations for physics tasks.

    Args:
        model: Base model from setup_lfm2_model
        r: LoRA rank (default 32)
        lora_alpha: LoRA alpha scaling (default 64)
        lora_dropout: Dropout probability (default 0.0)
        target_modules: Modules to apply LoRA to (default: attention + MLP)

    Returns:
        Model with LoRA adapters applied

    Raises:
        ImportError: If unsloth is not installed
    """
    if not _check_unsloth():
        raise ImportError(
            "unsloth is not installed. Install with: "
            "pip install git+https://github.com/unslothai/unsloth.git"
        )

    from unsloth import FastLanguageModel

    # Default target modules for transformer models
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",  # MLP
        ]

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    return model


def prepare_dataset(
    examples: List[Dict[str, str]],
    tokenizer: Any,
    max_length: int = 4096,
) -> Any:
    """
    Prepare examples for SFTTrainer.

    Args:
        examples: List of dicts with "input" and "output" keys
        tokenizer: Tokenizer from setup_lfm2_model
        max_length: Maximum tokenized length

    Returns:
        HuggingFace Dataset ready for SFTTrainer
    """
    from datasets import Dataset

    # Format examples as text for training
    formatted_examples = []
    for ex in examples:
        # Combine input and output with clear separator
        text = f"{ex['input']}\n\n{ex['output']}"
        formatted_examples.append({"text": text})

    dataset = Dataset.from_list(formatted_examples)
    return dataset


def train_stage(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    output_dir: str,
    config: Dict[str, Any],
) -> Any:
    """
    Train one curriculum stage using SFTTrainer.

    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        dataset: Prepared dataset from prepare_dataset
        output_dir: Directory for stage outputs
        config: Training config with keys:
            - epochs: Number of epochs (default 1)
            - batch_size: Per-device batch size (default 2)
            - gradient_accumulation_steps: Gradient accumulation (default 8)
            - learning_rate: Learning rate (default 2e-4)
            - max_seq_length: Maximum sequence length (default 4096)
            - warmup_ratio: Warmup fraction (default 0.1)
            - logging_steps: Steps between logs (default 10)

    Returns:
        SFTTrainer instance (for metrics access)

    Raises:
        ImportError: If trl is not installed
    """
    if not _check_trl():
        raise ImportError(
            "trl is not installed. Install with: pip install trl>=0.8.0"
        )

    from trl import SFTTrainer, SFTConfig

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract config with defaults
    epochs = config.get("epochs", 1)
    batch_size = config.get("batch_size", 2)
    grad_accum = config.get("gradient_accumulation_steps", 8)
    lr = config.get("learning_rate", 2e-4)
    max_seq_length = config.get("max_seq_length", 4096)
    warmup_ratio = config.get("warmup_ratio", 0.1)
    logging_steps = config.get("logging_steps", 10)
    save_steps = config.get("save_steps", 500)

    # Create training config
    training_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_seq_length=max_seq_length,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        bf16=True,  # Use bfloat16 for A6000
        optim="adamw_8bit",  # Memory-efficient optimizer
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        dataset_text_field="text",
        report_to="wandb",  # W&B integration
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_config,
    )

    # Train
    trainer.train()

    return trainer
