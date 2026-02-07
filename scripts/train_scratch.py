#!/usr/bin/env python3
"""Train GPT from scratch on physics prediction task.

This script trains a GPT model from scratch on physics simulation data
using digit-level tokenization, muP scaling for hyperparameter transfer,
and curriculum learning based on scene complexity.

Example usage:
    python scripts/train_scratch.py --data-dir data/train --model-size small
    python scripts/train_scratch.py --data-dir data/train --model-size medium --use-mup
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import project modules
from src.scratch.physics_tokenizer import PhysicsTokenizer
from src.scratch.model_config import (
    create_model_config,
    apply_mup_scaling,
    create_mup_optimizer,
    GPTConfig,
)
from src.scratch.gpt import GPT
from src.scratch.nanochat_adapter import PhysicsTask, PhysicsDataset
from src.training.curriculum import CurriculumDataset
from src.training.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
)
from src.training.wandb_utils import (
    setup_wandb,
    log_training_step,
    log_validation,
    log_curriculum_stage,
    finish_run,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GPT from scratch on physics prediction task."
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/train",
        help="Directory containing training JSONL files",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="data/val",
        help="Directory containing validation JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/gpt-physics",
        help="Directory to save checkpoints",
    )

    # Model arguments
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "medium"],
        default="small",
        help="Model size (small ~10M, medium ~85M params)",
    )
    parser.add_argument(
        "--use-mup",
        action="store_true",
        default=True,
        help="Use muP scaling for hyperparameter transfer",
    )
    parser.add_argument(
        "--no-mup",
        action="store_false",
        dest="use_mup",
        help="Disable muP scaling",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs per curriculum stage",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay coefficient",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )

    # Curriculum arguments
    parser.add_argument(
        "--curriculum-stages",
        type=int,
        default=5,
        help="Number of curriculum learning stages",
    )
    parser.add_argument(
        "--max-examples-per-stage",
        type=int,
        default=None,
        help="Maximum training examples per stage (for debugging)",
    )

    # Checkpointing and logging
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="Steps between validation evaluations",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Steps between checkpoint saves",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Steps between logging",
    )

    # W&B arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="physics-llm",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=None,
        help="W&B mode (auto-detected if not specified)",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )

    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    max_batches: Optional[int] = None,
) -> float:
    """Evaluate model on validation data.

    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        device: Device to use
        max_batches: Maximum batches to evaluate (for speed)

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        _, loss = model(input_ids, targets)
        total_loss += loss.item()
        num_batches += 1

        if max_batches and num_batches >= max_batches:
            break

    model.train()
    return total_loss / max(num_batches, 1)


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function to pad sequences to same length.

    Args:
        batch: List of samples from dataset

    Returns:
        Padded batch dictionary
    """
    # Find max length in batch
    max_len = max(sample["input_ids"].size(0) for sample in batch)

    # Pad sequences
    input_ids = []
    targets = []

    for sample in batch:
        seq_len = sample["input_ids"].size(0)
        pad_len = max_len - seq_len

        if pad_len > 0:
            # Pad with zeros (PAD token)
            input_ids.append(
                torch.cat([sample["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
            )
            # Pad targets with -1 (ignored in loss)
            targets.append(
                torch.cat([sample["targets"], torch.full((pad_len,), -1, dtype=torch.long)])
            )
        else:
            input_ids.append(sample["input_ids"])
            targets.append(sample["targets"])

    return {
        "input_ids": torch.stack(input_ids),
        "targets": torch.stack(targets),
    }


def main():
    """Main training function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup W&B
    config = vars(args)
    config["model_type"] = "gpt-scratch"
    run = setup_wandb(config, project=args.wandb_project, mode=args.wandb_mode)

    print(f"Training GPT-physics from scratch")
    print(f"  Model size: {args.model_size}")
    print(f"  Device: {args.device}")
    print(f"  muP: {args.use_mup}")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")

    # Setup tokenizer
    tokenizer = PhysicsTokenizer()
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Create model configuration
    model_config = create_model_config(args.model_size)
    model_config.vocab_size = tokenizer.vocab_size

    # Create model
    model = GPT(model_config, use_mup_scaling=args.use_mup)

    # Apply muP scaling if enabled
    if args.use_mup:
        base_config = create_model_config("small")
        base_config.vocab_size = tokenizer.vocab_size
        base_model = GPT(base_config, use_mup_scaling=False)
        model = apply_mup_scaling(model, base_model)
        optimizer = create_mup_optimizer(model, args.lr, args.weight_decay)
        print("  Using muP optimizer")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        print("  Using standard AdamW optimizer")

    # Move model to device
    model = model.to(args.device)

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            start_epoch, global_step, _ = load_checkpoint(
                args.resume, model, optimizer, device=args.device
            )
        else:
            print(f"Warning: Checkpoint not found: {args.resume}")
    else:
        # Check for latest checkpoint
        latest = find_latest_checkpoint(args.output_dir)
        if latest:
            print(f"Found latest checkpoint: {latest}")
            start_epoch, global_step, _ = load_checkpoint(
                latest, model, optimizer, device=args.device
            )

    # Setup curriculum
    print(f"\nSetting up curriculum with {args.curriculum_stages} stages...")
    curriculum = CurriculumDataset(args.data_dir)
    stages = curriculum.get_curriculum_stages(args.curriculum_stages)
    stage_info = curriculum.get_stage_info(args.curriculum_stages)

    for info in stage_info:
        print(
            f"  Stage {info['stage']}: {info['num_scenes']} scenes, "
            f"complexity {info['complexity_min']}-{info['complexity_max']}"
        )

    # Setup validation data (use first 100 scenes from val dir if exists)
    val_loader = None
    val_dir = Path(args.val_dir)
    if val_dir.exists():
        val_task = PhysicsTask(args.val_dir, tokenizer)
        val_scenes = list(val_dir.rglob("*.jsonl"))[:100]
        if val_scenes:
            val_data = val_task.get_val_data([str(p) for p in val_scenes])
            val_loader = DataLoader(
                val_data,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
            print(f"  Validation: {len(val_data)} examples from {len(val_scenes)} scenes")

    # Training loop
    print(f"\nStarting training...")
    model.train()
    best_val_loss = float("inf")

    for stage_idx, stage_paths in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"Curriculum Stage {stage_idx + 1}/{len(stages)}")
        print(f"{'='*60}")

        if not stage_paths:
            print("  No scenes in this stage, skipping...")
            continue

        # Log curriculum stage
        info = stage_info[stage_idx]
        log_curriculum_stage(
            stage_idx,
            info["num_scenes"],
            (info["complexity_min"], info["complexity_max"]),
        )

        # Limit paths if debugging
        if args.max_examples_per_stage:
            stage_paths = stage_paths[: args.max_examples_per_stage // 10]

        # Load stage data
        task = PhysicsTask(args.data_dir, tokenizer)
        train_data = task.get_train_data(stage_paths)

        if len(train_data) == 0:
            print("  No training examples, skipping...")
            continue

        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

        print(f"  Training on {len(train_data)} examples")

        # Train epochs on this stage
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(args.device)
                targets = batch["targets"].to(args.device)

                _, loss = model(input_ids, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                global_step += 1
                epoch_loss += loss.item()
                epoch_batches += 1

                # Logging
                if global_step % args.log_interval == 0:
                    log_training_step(global_step, loss.item(), args.lr)
                    print(f"  Step {global_step}: loss={loss.item():.4f}", flush=True)

                # Validation
                if val_loader and global_step % args.eval_interval == 0:
                    val_loss = evaluate(model, val_loader, args.device, max_batches=50)
                    log_validation(global_step, val_loss, {})
                    print(
                        f"  Step {global_step}: train_loss={loss.item():.4f}, "
                        f"val_loss={val_loss:.4f}"
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # Save best model
                        torch.save(
                            model.state_dict(),
                            output_dir / "best_model.pt",
                        )

                # Checkpointing
                if global_step % args.save_interval == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        None,  # No scheduler
                        epoch,
                        global_step,
                        loss.item(),
                        args.output_dir,
                        config=config,
                    )

            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            print(
                f"  Stage {stage_idx + 1} Epoch {epoch + 1}/{args.epochs}: "
                f"avg_loss={avg_epoch_loss:.4f}"
            )

    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete!")
    print(f"  Final model: {final_path}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Total steps: {global_step}")

    # Finish W&B run
    finish_run()


if __name__ == "__main__":
    main()
