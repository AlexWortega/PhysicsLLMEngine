#!/usr/bin/env python3
"""
Evaluation CLI for physics prediction models.

Provides command-line interface for running evaluation on trained models.

Usage:
    python scripts/run_evaluation.py --model finetune --checkpoint path/to/ckpt
    python scripts/run_evaluation.py --model scratch --checkpoint path/to/ckpt
    python scripts/run_evaluation.py --all --finetune-ckpt path/to/ft --scratch-ckpt path/to/scratch

Examples:
    # Evaluate fine-tuned model
    python scripts/run_evaluation.py --model finetune \\
        --checkpoint checkpoints/finetune/final \\
        --output-dir evaluation_results/finetune

    # Evaluate from-scratch model
    python scripts/run_evaluation.py --model scratch \\
        --checkpoint checkpoints/scratch/final \\
        --output-dir evaluation_results/scratch

    # Quick evaluation (fewer scenes, no OOD)
    python scripts/run_evaluation.py --model finetune \\
        --checkpoint checkpoints/finetune/final \\
        --num-scenes 100 --no-ood --num-gifs 3

    # Full evaluation with W&B
    python scripts/run_evaluation.py --all \\
        --finetune-ckpt checkpoints/finetune/final \\
        --scratch-ckpt checkpoints/scratch/final \\
        --wandb-project physics-prediction

    # OOD-only evaluation
    python scripts/run_evaluation.py --model finetune \\
        --checkpoint checkpoints/finetune/final \\
        --ood-only
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.runner import EvaluationRunner
from src.evaluation.report import generate_evaluation_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate physics prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["finetune", "scratch", "all"],
        help="Model type to evaluate: finetune (LFM2+LoRA), scratch (GPT), or all",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (for single model evaluation)",
    )
    parser.add_argument(
        "--finetune-ckpt",
        type=str,
        help="Path to fine-tuned model checkpoint (for --model all)",
    )
    parser.add_argument(
        "--scratch-ckpt",
        type=str,
        help="Path to from-scratch model checkpoint (for --model all)",
    )

    # Data paths
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/test",
        help="Test data directory (default: data/test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results (default: evaluation_results)",
    )

    # Evaluation settings
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=None,
        help="Number of scenes to evaluate (default: all)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=100,
        help="Maximum rollout steps (default: 100)",
    )
    parser.add_argument(
        "--num-gifs",
        type=int,
        default=10,
        help="Number of trajectory GIFs to generate (default: 10)",
    )
    parser.add_argument(
        "--ood-scenes",
        type=int,
        default=50,
        help="OOD scenes per type (default: 50)",
    )

    # Evaluation modes
    parser.add_argument(
        "--no-ood",
        action="store_true",
        help="Skip OOD evaluation",
    )
    parser.add_argument(
        "--ood-only",
        action="store_true",
        help="Run only OOD evaluation",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="physics-prediction-eval",
        help="W&B project name (default: physics-prediction-eval)",
    )

    # Model config (for from-scratch)
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "medium"],
        default="small",
        help="Model size for from-scratch (default: small)",
    )

    return parser.parse_args()


def load_finetune_model(checkpoint_path: str):
    """Load fine-tuned LFM2 model with LoRA adapters."""
    try:
        from src.finetune.lfm2_trainer import setup_lfm2_model
        from peft import PeftModel
        import torch

        print(f"Loading fine-tuned model from: {checkpoint_path}")

        # Load base model
        model, tokenizer = setup_lfm2_model()

        # Load LoRA adapters
        if Path(checkpoint_path).exists():
            model = PeftModel.from_pretrained(model, checkpoint_path)
            print("LoRA adapters loaded successfully")
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            print("Using base model without LoRA adapters")

        # Attach tokenizer for evaluation
        model.tokenizer = tokenizer
        model.eval()

        return model

    except ImportError as e:
        print(f"Error: Required packages not installed for fine-tuned model: {e}")
        print("Install with: pip install unsloth peft transformers")
        sys.exit(1)


def load_scratch_model(checkpoint_path: str, model_size: str = "small"):
    """Load from-scratch GPT model."""
    try:
        import torch
        from src.scratch.gpt import GPT
        from src.scratch.model_config import create_model_config
        from src.scratch.physics_tokenizer import PhysicsTokenizer

        print(f"Loading from-scratch model from: {checkpoint_path}")

        # Create config
        config = create_model_config(model_size)

        # Load checkpoint to get vocab_size
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            # Infer vocab_size from embedding shape
            if "transformer.wte.weight" in state_dict:
                vocab_size = state_dict["transformer.wte.weight"].shape[0]
                config.vocab_size = vocab_size
                print(f"Inferred vocab_size={vocab_size} from checkpoint")

        # Create model with correct vocab_size
        model = GPT(config)

        # Load checkpoint
        if Path(checkpoint_path).exists():
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print("Checkpoint loaded successfully")
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            print("Using randomly initialized model")

        # Attach tokenizer
        model.tokenizer = PhysicsTokenizer()
        model.eval()

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"Model loaded on {device}")

        return model

    except ImportError as e:
        print(f"Error: Required packages not installed for from-scratch model: {e}")
        sys.exit(1)


def run_ood_only_evaluation(args):
    """Run only OOD evaluation without loading full model."""
    from src.evaluation.ood_generator import generate_ood_scene, OOD_TYPES, get_ood_type_description
    import numpy as np

    print("\n" + "=" * 60)
    print("OOD Evaluation Mode (no model required)")
    print("=" * 60)

    print("\nOOD Types available:")
    for ood_type in OOD_TYPES:
        print(f"  - {ood_type}: {get_ood_type_description(ood_type)}")

    print(f"\nGenerating {args.ood_scenes} scenes per type...")

    # Just demonstrate scene generation (no model evaluation)
    for ood_type in OOD_TYPES:
        print(f"\n{ood_type}:")
        scenes = []
        for i in range(min(5, args.ood_scenes)):
            seed = 200000 + OOD_TYPES.index(ood_type) * 1000 + i
            scene = generate_ood_scene(seed, ood_type, ood_level=1.5, num_frames=10)
            scenes.append(scene)
            print(f"  Scene {i}: {scene['header']['num_objects']} objects, "
                  f"distance={scene['ood_distance']:.3f}")

    print("\nOOD scene generation complete!")
    print("To evaluate a model on OOD scenes, use --model flag with checkpoint.")


def main():
    """Main entry point."""
    args = parse_args()

    # Validate arguments
    if args.ood_only:
        run_ood_only_evaluation(args)
        return

    if not args.model:
        print("Error: --model is required (finetune, scratch, or all)")
        print("Use --help for usage information")
        sys.exit(1)

    if args.model in ["finetune", "scratch"] and not args.checkpoint:
        print(f"Error: --checkpoint is required for --model {args.model}")
        sys.exit(1)

    if args.model == "all" and not (args.finetune_ckpt and args.scratch_ckpt):
        print("Error: --finetune-ckpt and --scratch-ckpt are required for --model all")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize runner
    runner = EvaluationRunner(
        test_data_dir=args.test_dir,
        output_dir=str(output_dir),
        use_wandb=not args.no_wandb,
    )

    try:
        if args.model == "all":
            # Evaluate both models
            models = {}

            print("\nLoading fine-tuned model...")
            models["LFM2-LoRA"] = load_finetune_model(args.finetune_ckpt)

            print("\nLoading from-scratch model...")
            models[f"GPT-{args.model_size}"] = load_scratch_model(args.scratch_ckpt, args.model_size)

            # Evaluate all
            results = runner.evaluate_all_models(
                models,
                num_scenes=args.num_scenes,
                rollout_steps=args.rollout_steps,
                num_gifs=args.num_gifs,
                include_ood=not args.no_ood,
                ood_scenes_per_type=args.ood_scenes,
            )

        else:
            # Evaluate single model
            if args.model == "finetune":
                model = load_finetune_model(args.checkpoint)
                model_name = "LFM2-LoRA"
            else:
                model = load_scratch_model(args.checkpoint, args.model_size)
                model_name = f"GPT-{args.model_size}"

            results = runner.evaluate_model(
                model,
                model_name,
                num_scenes=args.num_scenes,
                rollout_steps=args.rollout_steps,
                num_gifs=args.num_gifs,
                include_ood=not args.no_ood,
                ood_scenes_per_type=args.ood_scenes,
            )

            # Generate single model report
            report_path = output_dir / "evaluation_report.md"
            generate_evaluation_report(
                {model_name: results},
                str(report_path),
                test_set_size=len(list(Path(args.test_dir).glob("**/*.jsonl"))) if Path(args.test_dir).exists() else 0,
                visualization_paths=results.get("visualization_paths", []),
            )
            print(f"\nReport generated: {report_path}")

        print("\n" + "=" * 60)
        print("Evaluation complete!")
        print(f"Results saved to: {output_dir}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
