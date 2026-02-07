#!/usr/bin/env python3
"""
Generate prediction examples on held-out validation set using latest LFM2 checkpoint.

Loads base LFM2-350M + LoRA adapter, runs inference on unseen + seen scenario types.
"""
import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.data_loader import load_physics_scene, _format_header_text, _format_frame_text

# Config
CHECKPOINT_DIR = "/home/alexw/checkpoints/lfm2-scenarios/stage3/checkpoint-1000"
VAL_DIR = "/home/alexw/data_scenarios/val"
OUTPUT_FILE = "/home/alexw/evaluation_results/heldout_examples_stage3.txt"

# Unseen types (never in training)
UNSEEN_TYPES = ["pong", "bowling", "ramp_roll", "angry_birds", "hourglass", "newtons_cradle",
                "planetary_rotation"]
# A few seen types for comparison
SEEN_TYPES = ["head_on", "tower", "pendulum", "projectile", "billiards", "dominos"]

SCENES_PER_TYPE = 2
CONTEXT_FRAMES = 10  # give model 10 frames of context
PREDICT_FRAMES = 5   # predict next 5 frames


def load_model():
    """Load LFM2 base model + LoRA adapter."""
    import torch
    from unsloth import FastLanguageModel
    from peft import PeftModel

    print("Loading base LFM2-350M model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="LiquidAI/LFM2-350M",
        max_seq_length=8192,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    print(f"Loading LoRA adapter from {CHECKPOINT_DIR}...")
    model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)
    model.eval()

    # Make sure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    print(f"Model loaded on {device}")
    return model, tokenizer


def generate_prediction(model, tokenizer, input_text, max_new_tokens=500):
    """Generate next frame prediction."""
    import torch

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=7000,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated


def get_scene_files(scenario_type, n=2):
    """Get n scene files for a given type."""
    type_dir = Path(VAL_DIR) / scenario_type
    if not type_dir.exists():
        return []

    scene_files = []
    for seed_dir in sorted(type_dir.iterdir()):
        if seed_dir.is_dir():
            for jsonl in sorted(seed_dir.glob("*.jsonl")):
                scene_files.append(str(jsonl))
                if len(scene_files) >= n:
                    return scene_files
    return scene_files


def format_ground_truth_frame(frame):
    """Format GT frame in the same style as model output for comparison."""
    lines = [f"Frame {frame['frame']}:"]
    for obj in frame['objects']:
        pos = obj['position']
        vel = obj['velocity']
        angle = obj.get('angle', 0)
        ang_vel = obj.get('angular_velocity', 0)
        obj_str = f"  obj_{obj['id']}: pos=({pos['x']:.4f}, {pos['y']:.4f}), vel=({vel['x']:.4f}, {vel['y']:.4f})"
        if abs(angle) > 0.001 or abs(ang_vel) > 0.001:
            obj_str += f", a={angle:.4f}, av={ang_vel:.4f}"
        lines.append(obj_str)
    return "\n".join(lines)


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    model, tokenizer = load_model()

    all_types = UNSEEN_TYPES + SEEN_TYPES
    results = []

    for scenario_type in all_types:
        is_unseen = scenario_type in UNSEEN_TYPES
        label = "UNSEEN" if is_unseen else "SEEN"
        print(f"\n{'='*60}")
        print(f"[{label}] {scenario_type}")
        print(f"{'='*60}")

        scene_files = get_scene_files(scenario_type, SCENES_PER_TYPE)
        if not scene_files:
            print(f"  No scenes found for {scenario_type}, skipping")
            continue

        for sf in scene_files:
            header, frames = load_physics_scene(sf)
            seed = header.get("seed", "?")
            print(f"  Scene seed={seed}, objects={header['object_count']}, "
                  f"difficulty={header.get('difficulty', '?')}, frames={len(frames)}")

            # Build context: header + first CONTEXT_FRAMES frames
            header_text = _format_header_text(header)
            context_text = header_text
            for f_idx in range(min(CONTEXT_FRAMES, len(frames))):
                context_text += _format_frame_text(frames[f_idx])

            # Predict next PREDICT_FRAMES frames autoregressively
            scene_result = {
                "type": scenario_type,
                "label": label,
                "seed": seed,
                "difficulty": header.get("difficulty", "?"),
                "objects": header["object_count"],
                "predictions": [],
                "ground_truths": [],
            }

            current_context = context_text
            for step in range(PREDICT_FRAMES):
                target_idx = CONTEXT_FRAMES + step
                if target_idx >= len(frames):
                    break

                gt_frame = frames[target_idx]

                # Generate prediction
                prompt = current_context + "Predict next frame:"
                pred_text = generate_prediction(model, tokenizer, prompt, max_new_tokens=300)

                gt_text = format_ground_truth_frame(gt_frame)

                scene_result["predictions"].append(pred_text.strip())
                scene_result["ground_truths"].append(gt_text)

                print(f"    Step {step+1} (frame {target_idx+1}): predicted")

                # Feed GT frame for next step (teacher forcing to avoid drift)
                current_context += _format_frame_text(gt_frame)

            results.append(scene_result)

    # Write results
    with open(OUTPUT_FILE, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LFM2-350M + LoRA (stage2/checkpoint-500) — Held-out Evaluation Examples\n")
        f.write(f"Checkpoint: {CHECKPOINT_DIR}\n")
        f.write(f"Context frames: {CONTEXT_FRAMES}, Predicted frames: {PREDICT_FRAMES}\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"\n{'='*70}\n")
            f.write(f"[{r['label']}] {r['type']} | seed={r['seed']} | "
                    f"difficulty={r['difficulty']} | objects={r['objects']}\n")
            f.write(f"{'='*70}\n\n")

            for i, (pred, gt) in enumerate(zip(r["predictions"], r["ground_truths"])):
                f.write(f"--- Step {i+1} ---\n")
                f.write(f"GROUND TRUTH:\n{gt}\n\n")
                f.write(f"PREDICTION:\n{pred}\n\n")

    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Total scenes: {len(results)}")


if __name__ == "__main__":
    main()
