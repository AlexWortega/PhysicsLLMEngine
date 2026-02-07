# Training & Evaluation Scripts

## Data Generation

### Generate Dataset

```bash
python scripts/generate_scenarios_dataset.py \
  --output-dir data_scenarios/train \
  --num-scenes-per-type 30000 \
  --scenario-types billiards bowling tower pendulum projectile \
  --num-workers 16 \
  --seed 42
```

**Options:**
- `--output-dir`: Where to save JSONL files (organized by scenario type)
- `--num-scenes-per-type`: Number of scenes per scenario type
- `--scenario-types`: Specific types to generate (default: all 35)
- `--num-workers`: Parallel workers (CPU cores)
- `--seed`: Random seed for reproducibility
- `--frames-per-scene`: Number of simulation frames (default: 200)
- `--difficulty-range`: Min and max difficulty (default: 1 5)

**Output Structure:**
```
data_scenarios/train/
├── billiards/
│   ├── scene_00000001.jsonl
│   ├── scene_00000002.jsonl
│   └── ...
├── bowling/
└── ...
```

### Generate Validation/Held-out Set

```bash
python scripts/generate_scenarios_dataset.py \
  --output-dir data_scenarios/val \
  --num-scenes-per-type 3000 \
  --scenario-types pong bowling ramp_roll angry_birds hourglass newtons_cradle \
  --seed 5000000
```

This creates validation scenes for held-out scenario types.

---

## Training

### Fine-tune LFM2 with LoRA

```bash
python scripts/train_finetune.py \
  --data-dir data_scenarios/train \
  --output-dir checkpoints/lfm2-scenarios \
  --curriculum-stages 5 \
  --epochs-per-stage 1 \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 2e-4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --max-seq-length 8192 \
  --wandb-project physics-llm
```

**Key Arguments:**
- `--curriculum-stages`: Number of difficulty stages (1-5)
- `--epochs-per-stage`: Training epochs per curriculum stage
- `--batch-size`: Per-device batch size
- `--grad-accum`: Gradient accumulation steps (effective batch = batch_size × grad_accum)
- `--lr`: Learning rate
- `--lora-r`: LoRA rank
- `--lora-alpha`: LoRA alpha scaling
- `--max-seq-length`: Context window (default: 8192)
- `--wandb-project`: Weights & Biases project name

**Curriculum Details:**

The script automatically:
1. Loads scenes filtered by difficulty
2. Trains 1 epoch per stage
3. Saves checkpoint after each stage
4. Logs to W&B

**Checkpoints:**
```
checkpoints/lfm2-scenarios/
├── stage0/
│   └── checkpoint-1000/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── ...
├── stage1/
└── ...
```

### Train GPT from Scratch (Baseline)

```bash
python scripts/train_scratch.py \
  --data-dir data_scenarios/train \
  --output-dir checkpoints/gpt-scratch \
  --model-size 350M \
  --batch-size 8 \
  --grad-accum 4 \
  --lr 3e-4 \
  --max-steps 100000
```

**Model Sizes:**
- `125M`: 12 layers, 768 dim, 12 heads
- `350M`: 24 layers, 1024 dim, 16 heads
- `760M`: 24 layers, 1536 dim, 16 heads

Uses muP scaling for stable training across model sizes.

---

## Evaluation

### Run Full Evaluation Pipeline

```bash
python scripts/run_evaluation.py \
  --model finetune \
  --checkpoint checkpoints/lfm2-scenarios/stage2/checkpoint-1000 \
  --val-dir data_scenarios/val \
  --output-dir evaluation_results \
  --context-frames 10 \
  --rollout-frames 20 \
  --max-scenes-per-type 100
```

**Arguments:**
- `--model`: `finetune` (LFM2+LoRA) or `scratch` (GPT baseline)
- `--checkpoint`: Path to saved checkpoint
- `--val-dir`: Validation data directory
- `--output-dir`: Where to save results
- `--context-frames`: Number of frames to condition on
- `--rollout-frames`: Number of frames to predict autoregressively
- `--max-scenes-per-type`: Limit evaluation scenes (for speed)

**Output:**
```
evaluation_results/
├── metrics.json          # MSE, accuracy per scenario type
├── predictions/          # Generated frame predictions
│   ├── pong_scene_00001.txt
│   └── ...
└── comparison_gifs/      # Side-by-side GT vs Prediction
    ├── pong_cmp.gif
    └── ...
```

### Render Comparison GIFs

```bash
python scripts/render_comparison_gifs.py \
  --checkpoint checkpoints/lfm2-scenarios/stage2/checkpoint-1000 \
  --val-dir data_scenarios/val \
  --output-dir evaluation_results/comparison_gifs \
  --scenario-types pong bowling angry_birds \
  --num-scenes 5
```

Creates side-by-side animations:
- **Left panel:** Ground Truth (green objects)
- **Right panel:** LLM Prediction (blue objects)

---

## Utilities

### Validate Dataset

```bash
python scripts/validate_dataset.py \
  --data-dir data_scenarios/train \
  --check-physics \
  --verbose
```

Checks:
- File format correctness
- Physics determinism (re-simulate and compare)
- Missing files
- Duplicate scenes

### Generate Held-out Examples

```bash
python scripts/generate_heldout_examples.py \
  --output heldout_examples.txt \
  --num-examples 10 \
  --scenario-types pong bowling
```

Creates text file with example scenes for manual inspection.

### Render Scenario Gallery

```bash
python scripts/render_gallery.py \
  --output-dir assets/scenario_gifs \
  --scenario-types all \
  --difficulty 3 \
  --fps 15
```

Generates individual GIF for each scenario type.

---

## Environment Setup

### Install Dependencies

```bash
# Core training
pip install torch transformers peft unsloth-zoo bitsandbytes

# Data generation
pip install pymunk numpy

# Visualization
pip install matplotlib pillow imageio

# Evaluation
pip install scikit-learn wandb

# Hugging Face upload
pip install huggingface_hub datasets
```

### GPU Requirements

| Task | VRAM | Recommended GPU |
|------|------|-----------------|
| Fine-tune LFM2-350M (LoRA) | 20-30 GB | RTX 3090, A6000 |
| Train GPT-350M from scratch | 40+ GB | A100 40GB |
| Evaluation (inference) | 16 GB | RTX 3080, V100 |
| Data generation (CPU) | N/A | 16+ cores recommended |

### Distributed Training

Use `accelerate` for multi-GPU:

```bash
accelerate config  # Configure distributed setup
accelerate launch scripts/train_finetune.py --batch-size 2 --grad-accum 16
```

---

## Troubleshooting

### Out of Memory

- Reduce `--batch-size` (increase `--grad-accum` to compensate)
- Reduce `--max-seq-length` (e.g., 4096 instead of 8192)
- Enable gradient checkpointing (already on by default)
- Use 8-bit optimizer (`--optimizer adamw_8bit`)

### Training Diverges

- Lower learning rate (`--lr 1e-4`)
- Increase warmup steps (`--warmup-steps 100`)
- Check curriculum stage (start from stage 0 if stage 2+ diverges)
- Verify dataset integrity with `validate_dataset.py`

### Slow Data Loading

- Increase `--num-workers` in dataloader (default: 4)
- Use SSD for dataset storage (not HDD)
- Pre-filter dataset by difficulty before training

### Evaluation Hangs

- Reduce `--max-scenes-per-type`
- Skip GIF rendering (`--no-render`)
- Run on CPU if GPU memory is exhausted

---

## Tips

- **Start small:** Train stage 0 (easy difficulty) first to verify setup
- **Monitor W&B:** Check loss curves for divergence or overfitting
- **Save checkpoints frequently:** Training can be interrupted
- **Use curriculum:** Stage-by-stage training stabilizes convergence
- **Test on held-out scenarios:** True test of generalization, not just validation loss
