#!/usr/bin/env python3
"""
Fine-tune LFM2-350M on physics prediction task with curriculum learning.

Uses Unsloth for efficient fine-tuning with LoRA adapters, curriculum-based
training through scenes of increasing complexity, and physics-informed
auxiliary loss for conservation constraints.

Example usage:
    # Full training run
    python scripts/train_finetune.py --data-dir data/train --output-dir checkpoints/lfm2-physics

    # Quick test with limited data
    python scripts/train_finetune.py --max-examples-per-stage 100 --epochs-per-stage 1

    # Resume from checkpoint
    python scripts/train_finetune.py --resume checkpoints/lfm2-physics

    # Offline mode (no W&B)
    python scripts/train_finetune.py --wandb-offline
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune LFM2-350M on physics prediction with curriculum learning"
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/train",
        help="Directory containing training scene JSONL files",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/lfm2-physics",
        help="Directory for checkpoints and outputs",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs-per-stage",
        type=int,
        default=1,
        help="Number of training epochs per curriculum stage",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length for tokenization",
    )

    # Curriculum arguments
    parser.add_argument(
        "--curriculum-stages",
        type=int,
        default=5,
        help="Number of curriculum stages (sorted by object count)",
    )
    parser.add_argument(
        "--max-examples-per-stage",
        type=int,
        default=None,
        help="Maximum examples per stage (for quick testing)",
    )
    parser.add_argument(
        "--max-context-frames",
        type=int,
        default=5,
        help="Maximum context frames for training examples",
    )
    parser.add_argument(
        "--complexity-metric",
        type=str,
        default="object_count",
        help="Header field for curriculum sorting (object_count or difficulty)",
    )

    # Physics loss arguments
    parser.add_argument(
        "--physics-loss-weight",
        type=float,
        default=0.01,
        help="Weight for physics auxiliary loss (0.01-0.1)",
    )

    # Resume arguments
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )

    # W&B arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="physics-llm",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run W&B in offline mode",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha scaling",
    )

    return parser.parse_args()


def load_stage_examples(
    stage_paths: List[str],
    max_examples: Optional[int],
    max_context_frames: int,
    max_total_tokens: int = 0,
) -> List[Dict[str, str]]:
    """
    Load training examples from stage scene files.

    Shuffles scene paths to ensure diverse sampling across scenario types
    when max_examples limits the total.

    Args:
        stage_paths: List of JSONL file paths for this stage
        max_examples: Maximum examples to load (None = unlimited)
        max_context_frames: Maximum context frames per example
        max_total_tokens: Token budget per example (0 = unlimited)

    Returns:
        List of dicts with "input" and "output" keys
    """
    import random
    from src.training import jsonl_to_training_examples

    # Shuffle to sample diverse scenarios instead of alphabetical order
    shuffled_paths = list(stage_paths)
    random.shuffle(shuffled_paths)

    examples = []

    for scene_path in shuffled_paths:
        try:
            for input_text, output_text, metadata in jsonl_to_training_examples(
                scene_path,
                max_context_frames=max_context_frames,
                max_total_tokens=max_total_tokens,
            ):
                examples.append({
                    "input": input_text,
                    "output": output_text,
                    "metadata": metadata,
                })

                if max_examples and len(examples) >= max_examples:
                    return examples
        except Exception as e:
            print(f"Warning: Failed to load {scene_path}: {e}")
            continue

    return examples


def save_training_checkpoint(
    model: Any,
    stage_idx: int,
    output_dir: str,
    config: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
) -> str:
    """
    Save training checkpoint after completing a stage.

    Args:
        model: Model to save
        stage_idx: Current stage index
        output_dir: Output directory
        config: Training configuration
        metrics: Training metrics

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(output_dir) / f"stage{stage_idx}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapters
    adapter_path = checkpoint_dir / "adapter"
    model.save_pretrained(str(adapter_path))

    # Save metadata
    metadata = {
        "stage": stage_idx,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics or {},
    }
    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved checkpoint to {checkpoint_dir}")
    return str(checkpoint_dir)


def find_latest_stage_checkpoint(checkpoint_dir: str) -> Optional[int]:
    """
    Find the latest completed stage from checkpoints.

    Args:
        checkpoint_dir: Directory containing stage checkpoints

    Returns:
        Latest completed stage index, or None if no checkpoints
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    # Find stage directories
    stage_dirs = list(checkpoint_path.glob("stage*"))
    if not stage_dirs:
        return None

    # Parse stage numbers and find max
    stages = []
    for d in stage_dirs:
        try:
            stage_num = int(d.name.replace("stage", ""))
            # Verify metadata exists (complete checkpoint)
            if (d / "metadata.json").exists():
                stages.append(stage_num)
        except ValueError:
            continue

    return max(stages) if stages else None


def main():
    """Main training entry point."""
    args = parse_args()

    print("=" * 60)
    print("LFM2-350M Fine-tuning with Curriculum Learning")
    print("=" * 60)

    # Build config
    config = vars(args).copy()
    config["model"] = "LiquidAI/LFM2-350M"
    config["timestamp"] = datetime.now().isoformat()

    # 1. Setup W&B
    print("\n[1/6] Setting up W&B...")
    from src.training import setup_wandb, log_curriculum_stage, save_checkpoint_artifact

    mode = "offline" if args.wandb_offline else None
    run = setup_wandb(config, project=args.wandb_project, mode=mode)
    if run:
        print(f"W&B run: {run.name}")
    else:
        print("W&B disabled or not available")

    # 2. Setup model with LoRA
    print("\n[2/6] Setting up LFM2-350M with LoRA adapters...")
    from src.finetune import setup_lfm2_model, add_lora_adapters, prepare_dataset, train_stage

    model, tokenizer = setup_lfm2_model(
        max_seq_length=args.max_seq_length,
    )
    model = add_lora_adapters(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    print(f"Model loaded: LFM2-350M + LoRA (r={args.lora_r}, alpha={args.lora_alpha})")

    # 3. Setup curriculum
    print("\n[3/6] Setting up curriculum...")
    from src.training import CurriculumDataset

    curriculum = CurriculumDataset(args.data_dir, complexity_metric=args.complexity_metric)
    stages = curriculum.get_curriculum_stages(args.curriculum_stages)
    stage_info = curriculum.get_stage_info(args.curriculum_stages)

    print(f"Total scenes: {len(curriculum)}")
    print(f"Curriculum stages: {args.curriculum_stages}")
    for info in stage_info:
        print(f"  Stage {info['stage']}: {info['num_scenes']} scenes "
              f"(objects: {info['complexity_min']}-{info['complexity_max']})")

    # 4. Resume handling
    print("\n[4/6] Checking for resume...")
    start_stage = 0
    if args.resume:
        latest_stage = find_latest_stage_checkpoint(args.resume)
        if latest_stage is not None:
            start_stage = latest_stage + 1
            print(f"Resuming from stage {start_stage} (completed stages 0-{latest_stage})")

            # Load adapter weights from latest checkpoint
            adapter_path = Path(args.resume) / f"stage{latest_stage}" / "adapter"
            if adapter_path.exists():
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(adapter_path))
                print(f"Loaded adapter weights from {adapter_path}")
        else:
            print("No checkpoints found, starting from scratch")
    else:
        print("Starting fresh training")

    # 5. Train through curriculum stages
    print("\n[5/6] Training through curriculum stages...")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_stages = len(stages)
    for stage_idx in range(start_stage, total_stages):
        stage_paths = stages[stage_idx]
        info = stage_info[stage_idx]

        print(f"\n{'=' * 40}")
        print(f"Stage {stage_idx + 1}/{total_stages}")
        print(f"Scenes: {len(stage_paths)}")
        print(f"Object count: {info['complexity_min']}-{info['complexity_max']}")
        print("=" * 40)

        # Log curriculum stage to W&B
        log_curriculum_stage(
            stage_idx,
            len(stage_paths),
            (info['complexity_min'], info['complexity_max']),
        )

        # Load stage data
        print("Loading examples...")
        examples = load_stage_examples(
            stage_paths,
            args.max_examples_per_stage,
            args.max_context_frames,
            max_total_tokens=args.max_seq_length,
        )
        print(f"Loaded {len(examples)} training examples")

        if not examples:
            print("Warning: No examples loaded, skipping stage")
            continue

        # Prepare dataset
        dataset = prepare_dataset(examples, tokenizer, args.max_seq_length)

        # Adjust LR for later stages (decay by 10% per stage)
        stage_lr = args.lr * (0.9 ** stage_idx)
        print(f"Learning rate: {stage_lr:.6f}")

        # Training config for this stage
        stage_config = {
            "epochs": args.epochs_per_stage,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "learning_rate": stage_lr,
            "max_seq_length": args.max_seq_length,
            "warmup_ratio": 0.1,
            "logging_steps": 10,
            "save_steps": 500,
        }

        # Train stage
        print("Training...")
        stage_output_dir = str(output_path / f"stage{stage_idx}")
        trainer = train_stage(
            model, tokenizer, dataset,
            output_dir=stage_output_dir,
            config=stage_config,
        )

        # Get training metrics
        metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}

        # Save checkpoint
        checkpoint_path = save_training_checkpoint(
            model, stage_idx, args.output_dir,
            config=config,
            metrics=metrics,
        )

        # Log artifact to W&B
        save_checkpoint_artifact(
            f"{checkpoint_path}/adapter/adapter_model.safetensors",
            epoch=stage_idx,
            step=trainer.state.global_step if hasattr(trainer.state, 'global_step') else 0,
            metadata={"stage": stage_idx, **metrics},
        )

        print(f"Stage {stage_idx + 1} complete!")

    # 6. Save final model
    print("\n[6/6] Saving final model...")
    final_path = output_path / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print(f"\nTraining complete!")
    print(f"Final model: {final_path}")

    # Finish W&B run
    from src.training import finish_run
    finish_run()


if __name__ == "__main__":
    main()
