"""Nanochat adapter for physics prediction task.

Provides PhysicsTask class for nanochat training interface and
model setup utilities with muP scaling support.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset

from src.training.data_loader import jsonl_to_training_examples


class PhysicsDataset(Dataset):
    """PyTorch Dataset for physics training examples."""

    def __init__(
        self,
        examples: List[tuple],
        tokenizer: Any,
        max_length: int = 1024,
    ) -> None:
        """Initialize dataset with examples.

        Args:
            examples: List of (input_text, output_text, metadata) tuples
            tokenizer: PhysicsTokenizer instance
            max_length: Maximum sequence length (truncate if longer)
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example.

        Returns dict with:
        - input_ids: Tokenized input (context frames)
        - targets: Tokenized target (next frame prediction)
        """
        input_text, output_text, metadata = self.examples[idx]

        # Tokenize input and output
        input_ids = self.tokenizer.encode(input_text)
        target_ids = self.tokenizer.encode(output_text)

        # Combine for language modeling: input + output
        # Target is shifted by 1 (next token prediction)
        combined = input_ids + target_ids

        # Truncate if too long
        if len(combined) > self.max_length:
            combined = combined[: self.max_length]

        # Create input and target tensors
        # For LM, target is the same sequence shifted by 1
        input_tensor = torch.tensor(combined[:-1], dtype=torch.long)
        target_tensor = torch.tensor(combined[1:], dtype=torch.long)

        return {
            "input_ids": input_tensor,
            "targets": target_tensor,
        }


class PhysicsTask:
    """Task interface for nanochat training on physics data.

    Provides methods to load training and validation data in the format
    expected by nanochat training infrastructure.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: Any,
        max_length: int = 1024,
        min_context_frames: int = 1,
        max_context_frames: Optional[int] = None,
    ) -> None:
        """Initialize physics task.

        Args:
            data_dir: Directory containing scene JSONL files (or list of paths)
            tokenizer: PhysicsTokenizer instance
            max_length: Maximum sequence length
            min_context_frames: Minimum context frames for training examples
            max_context_frames: Maximum context frames (None = no limit)
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_context_frames = min_context_frames
        self.max_context_frames = max_context_frames
        self._train_examples: Optional[List[tuple]] = None
        self._val_examples: Optional[List[tuple]] = None

    def _load_examples(self, paths: List[str]) -> List[tuple]:
        """Load training examples from scene files.

        Args:
            paths: List of JSONL file paths

        Returns:
            List of (input_text, output_text, metadata) tuples
        """
        examples = []
        total = len(paths)
        for i, path in enumerate(paths):
            if i % 50 == 0:
                print(f"    Loading scenes: {i}/{total} ({100*i//total}%)", flush=True)
            if not Path(path).exists():
                continue
            try:
                for example in jsonl_to_training_examples(
                    path,
                    min_context_frames=self.min_context_frames,
                    max_context_frames=self.max_context_frames,
                ):
                    examples.append(example)
            except Exception:
                # Skip invalid files
                continue
        print(f"    Loading scenes: {total}/{total} (100%) - done", flush=True)
        return examples

    def get_train_data(self, scene_paths: Optional[List[str]] = None) -> PhysicsDataset:
        """Get training dataset.

        Args:
            scene_paths: Optional list of specific scene paths to use.
                        If None, scans data_dir for all JSONL files.

        Returns:
            PhysicsDataset for training
        """
        if scene_paths is not None:
            paths = scene_paths
        elif isinstance(self.data_dir, list):
            paths = self.data_dir
        else:
            # Scan directory for JSONL files
            data_path = Path(self.data_dir)
            if data_path.exists():
                paths = [str(p) for p in data_path.rglob("*.jsonl")]
            else:
                paths = []

        examples = self._load_examples(paths)
        return PhysicsDataset(examples, self.tokenizer, self.max_length)

    def get_val_data(self, scene_paths: Optional[List[str]] = None) -> PhysicsDataset:
        """Get validation dataset.

        Args:
            scene_paths: Optional list of specific scene paths to use.

        Returns:
            PhysicsDataset for validation
        """
        # Same logic as train data
        return self.get_train_data(scene_paths)


def setup_nanochat_model(
    config: Any,
    tokenizer: Any,
    use_mup: bool = True,
) -> torch.nn.Module:
    """Set up GPT model for nanochat training.

    Args:
        config: GPTConfig instance
        tokenizer: PhysicsTokenizer instance
        use_mup: Whether to apply muP scaling

    Returns:
        GPT model ready for training
    """
    # Import here to avoid circular dependency
    from src.scratch.gpt import GPT
    from src.scratch.model_config import apply_mup_scaling, create_model_config

    # Set vocab size from tokenizer
    config.vocab_size = tokenizer.vocab_size

    # Create model
    model = GPT(config)

    if use_mup:
        # Create base model for muP scaling
        base_config = create_model_config("small")
        base_config.vocab_size = tokenizer.vocab_size
        base_model = GPT(base_config)
        model = apply_mup_scaling(model, base_model)

    return model
