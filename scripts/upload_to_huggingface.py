#!/usr/bin/env python3
"""
Upload trained model and dataset to Hugging Face Hub.

Requirements:
    pip install huggingface_hub

Usage:
    # Upload model adapter
    python scripts/upload_to_huggingface.py --type model \
        --checkpoint checkpoints/lfm2-scenarios/stage2/checkpoint-1000 \
        --repo alexwortega/LFM_Physics350M
    
    # Upload dataset
    python scripts/upload_to_huggingface.py --type dataset \
        --data-dir data_scenarios \
        --repo alexwortega/physics-scenarios
"""
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

def upload_model(checkpoint_path: str, repo_id: str, token: str = None):
    """Upload LoRA adapter to Hugging Face Hub."""
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📦 Uploading model from {checkpoint_path} to {repo_id}")
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
        print(f"✅ Repository {repo_id} ready")
    except Exception as e:
        print(f"⚠️  Repository creation: {e}")
    
    # Upload adapter files
    api = HfApi()
    
    # Files to upload
    files_to_upload = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "README.md",  # Will create if missing
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    
    # Create README if missing
    readme_path = checkpoint_path / "README.md"
    if not readme_path.exists():
        model_card_source = Path(__file__).parent.parent / "MODEL_CARD.md"
        if model_card_source.exists():
            with open(model_card_source) as f:
                readme_content = f.read()
            with open(readme_path, "w") as f:
                f.write(readme_content)
            print("✅ Created README.md from MODEL_CARD.md")
    
    # Upload files
    for filename in files_to_upload:
        file_path = checkpoint_path / filename
        if file_path.exists():
            print(f"  Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
        else:
            print(f"  ⚠️  Skipping {filename} (not found)")
    
    print(f"✅ Model uploaded to https://huggingface.co/{repo_id}")

def upload_dataset(data_dir: str, repo_id: str, token: str = None):
    """Upload physics scenarios dataset to Hugging Face Hub."""
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"📦 Uploading dataset from {data_dir} to {repo_id}")
    
    # Create repo
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
        print(f"✅ Repository {repo_id} ready")
    except Exception as e:
        print(f"⚠️  Repository creation: {e}")
    
    # Create dataset card
    dataset_card = f"""---
language:
- en
license: mit
task_categories:
- text-generation
tags:
- physics
- simulation
- rigid-body-dynamics
size_categories:
- 100K<n<1M
---

# Physics Scenarios Dataset

**1 million 2D rigid body physics simulation scenes for training LLMs to predict physics.**

This dataset contains 900,000 training scenes and 100,020 validation scenes across 35 scenario types (billiards, bowling, towers, pendulums, angry birds, etc.). Each scene has 200 frames of object positions, velocities, and metadata in structured text format.

## Dataset Structure

```
data_scenarios/
├── train/               # 900K scenes (24 scenario types)
│   ├── billiards/
│   ├── bowling/
│   └── ...
└── val/                 # 100K scenes (35 types, including 6 held-out)
    ├── pong/
    ├── angry_birds/
    └── ...
```

Each scene is a `.jsonl` file with:
- **1 header line:** Scene configuration (objects, gravity, scenario type, difficulty)
- **200 frame lines:** Object states (position, velocity, angle, angular velocity)

## Format

```
Scene: 3 objects (2 circles, 1 rectangle). Gravity: (0.00, -981.00). Timestep: 0.0167.
Scenario: pong (minigame), difficulty: 4

Objects:
  obj_0: circle r=10.0 at (400.00, 300.00), density=1.0, friction=0.5
  obj_1: circle r=15.0 at (200.00, 500.00), density=1.0, friction=0.5
  obj_2: rect 80x10 at (400.00, 50.00), density=2.0, friction=0.8

Frame 1: 3 of 3 objects moving.
  obj_0: pos=(405.12, 298.37), vel=(307.14, -97.81)
  obj_1: pos=(195.23, 502.45), vel=(-286.22, 147.58)
  ...
```

## Scenarios

35 scenario types across 6 categories:

| Category | Count | Types |
|----------|-------|-------|
| Collision | 5 | billiards, bowling, explosion, head_on, projectile |
| Stacking | 5 | tower, pyramid, dominos, jenga, bridge |
| Ramp | 5 | ramp_roll, ski_jump, funnel, plinko, marble_run |
| Constraint | 5 | pendulum, newtons_cradle, wrecking_ball, chain, seesaw |
| Minigame | 5 | angry_birds, pinball, basketball, breakout, pong |
| Complex | 10 | avalanche, conveyor, hourglass, orbit, wind, particle_explosion, chain_reaction, fluid_sim, solar_system, gravity_well |

## Statistics

- **Total scenes:** 1,000,020
- **Total frames:** 200,004,000 (~200M)
- **Held-out types:** 6 (pong, bowling, ramp_roll, angry_birds, hourglass, newtons_cradle)
- **Size:** ~582 GB (uncompressed JSONL)

## Generation

All scenes were generated deterministically using [Pymunk](http://www.pymunk.org/) (Python wrapper for Chipmunk2D physics engine). See [PhysicsLLMEngine](https://github.com/AlexWortega/PhysicsLLMEngine) for generation code.

## License

MIT

## Citation

```bibtex
@dataset{{physics_scenarios,
  title={{Physics Scenarios: 1M 2D Rigid Body Simulations for LLM Training}},
  author={{Wortega, Alex}},
  year={{2026}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```
"""
    
    readme_path = data_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(dataset_card)
    print("✅ Created dataset README.md")
    
    # Upload folder
    api = HfApi()
    
    # Note: For large datasets, consider uploading splits separately
    # or using HF datasets library with push_to_hub
    print("⚠️  Large dataset upload — this may take hours!")
    print("  Consider using:")
    print("    from datasets import Dataset")
    print("    dataset.push_to_hub(...)")
    
    upload_folder(
        folder_path=str(data_dir),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        ignore_patterns=["*.pyc", "__pycache__", ".git"],
    )
    
    print(f"✅ Dataset uploaded to https://huggingface.co/datasets/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Upload model or dataset to Hugging Face")
    parser.add_argument("--type", required=True, choices=["model", "dataset"],
                       help="What to upload")
    parser.add_argument("--checkpoint", help="Path to model checkpoint (for type=model)")
    parser.add_argument("--data-dir", help="Path to dataset directory (for type=dataset)")
    parser.add_argument("--repo", required=True, help="Hugging Face repo ID (e.g., alexwortega/LFM_Physics350M)")
    parser.add_argument("--token", help="Hugging Face API token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Get token from env if not provided
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("⚠️  No token provided. Public uploads will fail.")
        print("   Set --token or HF_TOKEN environment variable")
    
    if args.type == "model":
        if not args.checkpoint:
            parser.error("--checkpoint required for type=model")
        upload_model(args.checkpoint, args.repo, token)
    
    elif args.type == "dataset":
        if not args.data_dir:
            parser.error("--data-dir required for type=dataset")
        upload_dataset(args.data_dir, args.repo, token)

if __name__ == "__main__":
    main()
