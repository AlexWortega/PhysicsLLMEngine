---
language:
- en
license: mit
base_model: LiquidAI/LFM2-350M
tags:
- physics
- simulation
- lora
- text-generation
- rigid-body-dynamics
library_name: transformers
pipeline_tag: text-generation
datasets:
- alexwortega/physics-scenarios
---

# LFM_Physics350M

**A 350M-parameter language model fine-tuned to predict 2D rigid body physics from text.**

This model is a LoRA fine-tune of [LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) trained on 900,000 physics simulation scenes across 35 scenario types. Given a textual description of object positions, velocities, and scene configuration, the model autoregressively predicts the next frame of a physics simulation — effectively learning Newtonian mechanics through next-token prediction.

## Model Description

- **Base Model:** LiquidAI/LFM2-350M (Liquid Foundation Model)
- **Fine-tuning Method:** LoRA (rank 32, alpha 64)
- **Training Data:** 900K scenes, 180M frames from Pymunk 2D physics engine
- **Scenarios:** 35 physics scenario types (billiards, bowling, towers, pendulums, angry birds, etc.)
- **Task:** Autoregressive next-frame prediction from structured text
- **Precision:** bfloat16
- **Context Window:** 8192 tokens

## Training

### Curriculum Learning

The model was trained using a 5-stage curriculum from simple 2-object scenes to chaotic 50+ object systems:

| Stage | Difficulty | Objects | Examples | Loss |
|-------|-----------|---------|----------|------|
| 0 | Easy | 2-5 | 50,000 | 0.562 |
| 1 | Moderate | 5-15 | 50,000 | 0.609 |
| 2 | Complex | 10-30 | 50,000 | 0.622 |
| 3 | Hard | 20-40 | 50,000 | (training) |
| 4 | Extreme | 30-50+ | 50,000 | (pending) |

### Hyperparameters

```python
{
  "model": "LiquidAI/LFM2-350M",
  "lora_r": 32,
  "lora_alpha": 64,
  "lora_dropout": 0.05,
  "batch_size": 4,
  "gradient_accumulation": 8,
  "learning_rate": 2e-4,
  "optimizer": "adamw_8bit",
  "max_seq_length": 8192,
  "warmup_steps": 10,
  "num_epochs_per_stage": 1
}
```

### Infrastructure

- **GPU:** NVIDIA RTX A6000 (48 GB VRAM)
- **Training Time:** ~7-14 hours per curriculum stage
- **Framework:** Unsloth + Transformers
- **Logging:** Weights & Biases

## Performance

### Zero-Shot Generalization

The model was evaluated on **6 held-out scenario types never seen during training**:

- `pong` (minigame)
- `bowling` (collision)
- `ramp_roll` (ramp)
- `angry_birds` (complex minigame)
- `hourglass` (chaotic flow)
- `newtons_cradle` (constraint physics)

**Results: Sub-pixel accuracy on unseen scenarios**

Example prediction on unseen `pong` scenario (800x600 canvas):

```
GROUND TRUTH:  obj_0: pos=(339.83, 143.25), vel=(-531.04, -320.00)
PREDICTION:    obj_0: pos=(339.79, 142.05), vel=(-531.04, -320.00)
                           Δx=0.05      Δy=1.20       velocity: exact
```

- **Positional error:** <1 pixel (<0.2% on canvas)
- **Velocity error:** Near-zero (model learns exact linear dynamics)
- **Multi-body:** Handles 11+ objects with collision cascades

## Input Format

The model expects structured text describing physics scenes:

```
Scene: 3 objects (2 circles, 1 rectangle). Gravity: (0.00, -981.00). Timestep: 0.0167.
Scenario: pong (minigame), difficulty: 4

Objects:
  obj_0: circle r=10.0 at (400.00, 300.00), density=1.0, friction=0.5
  obj_1: circle r=15.0 at (200.00, 500.00), density=1.0, friction=0.5
  obj_2: rect 80x10 at (400.00, 50.00), density=2.0, friction=0.8

Static geometry: 4 segments (walls)
Constraints: none

Frame 1: 3 of 3 objects moving.
  obj_0: pos=(405.12, 298.37), vel=(307.14, -97.81)
  obj_1: pos=(195.23, 502.45), vel=(-286.22, 147.58)
  obj_2: pos=(400.00, 50.00), vel=(0.00, 0.00)
```

Given frames 1..N, the model predicts frame N+1 autoregressively.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2-350M",
    torch_dtype="bfloat16",
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "alexwortega/LFM_Physics350M")
tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")

# Prepare input (scene description + context frames)
prompt = """Scene: 2 objects (1 circle, 1 rectangle). Gravity: (0.00, -981.00). Timestep: 0.0167.
Scenario: projectile (collision), difficulty: 2

Objects:
  obj_0: circle r=15.0 at (100.00, 400.00), density=1.0, friction=0.3
  obj_1: rect 50x50 at (600.00, 100.00), density=2.0, friction=0.5

Static geometry: 4 segments (walls)
Constraints: none

Frame 1: 1 of 2 objects moving.
  obj_0: pos=(105.00, 395.00), vel=(300.00, -100.00)
  obj_1: pos=(600.00, 100.00), vel=(0.00, 0.00)

Frame 2:"""

# Generate next frame
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1)
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(prediction)
```

## Scenarios Supported

The model understands 35 physics scenario types across 6 categories:

| Category | Scenarios |
|----------|-----------|
| **Collision** | billiards, bowling, explosion, head_on, projectile |
| **Stacking** | tower, pyramid, dominos, jenga, bridge |
| **Ramp** | ramp_roll, ski_jump, funnel, plinko, marble_run |
| **Constraint** | pendulum, newtons_cradle, wrecking_ball, chain, seesaw |
| **Minigame** | angry_birds, pinball, basketball, breakout, pong |
| **Complex** | avalanche, conveyor, hourglass, orbit, wind, particle_explosion, chain_reaction, fluid_sim, solar_system |

See [scenario gallery](https://github.com/AlexWortega/PhysicsLLMEngine#scenario-zoo) for GIFs.

## Limitations

- **2D only:** The model is trained exclusively on 2D rigid body physics
- **Deterministic physics:** Assumes fixed timestep, no randomness in simulation
- **Text-based:** No vision encoder — requires structured text input
- **Autoregressive drift:** Long rollouts (100+ frames) accumulate error
- **No soft-body or fluids:** Only rigid circles/rectangles with Pymunk constraints
- **Training still in progress:** Current checkpoint is from Stage 2 (difficulty 3)

## Training Data

The model was trained on the [alexwortega/physics-scenarios](https://huggingface.co/datasets/alexwortega/physics-scenarios) dataset:

- **900,000 training scenes** (24 scenario types)
- **100,020 validation scenes** (all 35 types including 6 held-out)
- **200 frames per scene** (180M total frames)
- **~582 GB** of structured text data

Each scene is deterministically generated with Pymunk (Python wrapper for Chipmunk2D physics engine).

## Ethical Considerations

This model simulates rigid body physics and has no direct ethical implications. Potential misuse scenarios are limited given the domain (2D physics simulation). The model does not process real-world data or make decisions affecting individuals.

## Citation

```bibtex
@software{lfm_physics350m,
  title={LFM_Physics350M: Learning Rigid Body Dynamics via Next-Token Prediction},
  author={Wortega, Alex},
  year={2026},
  url={https://huggingface.co/alexwortega/LFM_Physics350M}
}
```

## License

MIT License. See [LICENSE](https://github.com/AlexWortega/PhysicsLLMEngine/blob/main/LICENSE) for details.

## Links

- **GitHub:** [AlexWortega/PhysicsLLMEngine](https://github.com/AlexWortega/PhysicsLLMEngine)
- **Dataset:** [alexwortega/physics-scenarios](https://huggingface.co/datasets/alexwortega/physics-scenarios)
- **Base Model:** [LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M)
- **Framework:** [Unsloth](https://github.com/unslothai/unsloth)
