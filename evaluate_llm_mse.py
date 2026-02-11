#!/usr/bin/env python3
"""
Evaluate trained LLM vs Ground Truth using MSE.

Loads LFM2-350M + LoRA checkpoint, generates predictions using the exact
text format from training (data_loader.py), compares against Pymunk ground truth.
"""

import sys
import json
import re
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.physics.scenario_generator import generate_scenario
from src.data.formats import format_scene_header
from src.training.data_loader import _format_header_text, _format_frame_text


# Regex to parse obj_X: pos=(x, y), vel=(vx, vy) from generated text
OBJ_POS_VEL = re.compile(
    r"obj_(\d+):\s*pos=\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)"
    r"(?:,\s*vel=\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\))?"
)


def load_model(checkpoint_path="/home/alexw/checkpoints/lfm2-scenarios/final"):
    """Load LFM2 + LoRA model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model LiquidAI/LFM2-350M ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM2-350M",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {checkpoint_path} ...")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("Model loaded successfully")
    return model, tokenizer


def _sim_get_frame_dict(sim):
    """Get current simulation state as a frame dict matching data_loader format."""
    state = sim.get_state()

    # Add 'description' field matching format_frame in formats.py
    VELOCITY_THRESHOLD = 1.0
    moving = sum(
        1 for obj in state['objects']
        if abs(obj['velocity']['x']) > VELOCITY_THRESHOLD
        or abs(obj['velocity']['y']) > VELOCITY_THRESHOLD
    )
    total = len(state['objects'])
    if moving == 0:
        motion_desc = "All objects are at rest"
    elif moving == total:
        motion_desc = "All objects are in motion"
    else:
        motion_desc = f"{moving} of {total} objects are moving"

    state['description'] = f"Frame {state['frame']}: {motion_desc}."
    return state


def _build_context_prompt(header_dict, frame_dicts):
    """
    Build the prompt text using the exact same format as training data.

    Uses _format_header_text and _format_frame_text from data_loader.py
    so the model sees text identical to its training distribution.
    """
    header_text = _format_header_text(header_dict)
    frame_texts = [_format_frame_text(f) for f in frame_dicts]
    return header_text + "".join(frame_texts) + "Predict next frame:"


def extract_predicted_frame(generated_text, prompt_len_chars, num_objects):
    """
    Extract the first predicted frame from generated text after the prompt.

    Returns dict: {obj_id: [x, y]} for positions found.
    """
    # Get only the newly generated portion
    new_text = generated_text[prompt_len_chars:]

    positions = {}
    for match in OBJ_POS_VEL.finditer(new_text):
        obj_id = int(match.group(1))
        x = float(match.group(2))
        y = float(match.group(3))
        positions[obj_id] = [x, y]

    return positions


def _frame_dict_from_prediction(pred_positions, prev_frame_dict, frame_num):
    """
    Build a pseudo frame_dict from predicted positions for autoregressive feeding.

    Uses previous frame's velocity/angle as approximation since we primarily
    predict positions.
    """
    objects = []
    for obj in prev_frame_dict["objects"]:
        obj_id = obj["id"]
        if obj_id in pred_positions:
            pos = pred_positions[obj_id]
        else:
            pos = [obj["position"]["x"], obj["position"]["y"]]

        objects.append({
            "id": obj_id,
            "type": obj.get("type", "unknown"),
            "position": {"x": round(pos[0], 4), "y": round(pos[1], 4)},
            "velocity": obj["velocity"],  # carry forward
            "angle": obj.get("angle", 0),
            "angular_velocity": obj.get("angular_velocity", 0),
            "material": obj.get("material", {}),
        })

    moving = sum(
        1 for o in objects
        if abs(o["velocity"]["x"]) > 1.0 or abs(o["velocity"]["y"]) > 1.0
    )
    if moving == 0:
        desc = "All objects are at rest"
    elif moving == len(objects):
        desc = "All objects are in motion"
    else:
        desc = f"{moving} of {len(objects)} objects are moving"

    return {
        "frame": frame_num,
        "description": f"Frame {frame_num}: {desc}.",
        "objects": objects,
    }


def generate_one_frame(model, tokenizer, prompt_text):
    """Generate one frame of output from the model given a prompt."""
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text


def evaluate_scenario(model, tokenizer, scenario_type, difficulty, seed,
                      context_frames=10, predict_frames=5):
    """
    Evaluate LLM prediction vs Ground Truth for one scenario.

    Generates autoregressive predictions for predict_frames steps
    and computes per-step and average MSE.
    """
    # Generate scenario
    sim, metadata = generate_scenario(
        seed=seed, scenario_type=scenario_type, difficulty=difficulty
    )

    # Build header dict (same as training pipeline)
    header_dict = format_scene_header(sim, seed, metadata=metadata)

    # Simulate context_frames + predict_frames and capture frame dicts
    all_frame_dicts = []
    for _ in range(context_frames + predict_frames):
        sim.step()
        all_frame_dicts.append(_sim_get_frame_dict(sim))

    context_dicts = all_frame_dicts[:context_frames]
    gt_dicts = all_frame_dicts[context_frames:]

    num_objects = len(gt_dicts[0]["objects"])

    # Autoregressive prediction loop
    per_step_mse = []
    current_context = list(context_dicts)  # will grow with predictions

    for step_idx in range(predict_frames):
        gt_frame = gt_dicts[step_idx]

        # Build prompt
        prompt_text = _build_context_prompt(header_dict, current_context)
        prompt_len = len(prompt_text)

        # Generate
        full_text = generate_one_frame(model, tokenizer, prompt_text)

        # Extract predicted positions
        pred_pos = extract_predicted_frame(full_text, prompt_len, num_objects)

        # Build position arrays for MSE
        gt_positions = np.array([
            [obj["position"]["x"], obj["position"]["y"]]
            for obj in gt_frame["objects"]
        ])

        pred_positions = np.zeros_like(gt_positions)
        for obj in gt_frame["objects"]:
            oid = obj["id"]
            if oid in pred_pos:
                pred_positions[oid] = pred_pos[oid]
            else:
                # Fallback: last known position (penalized)
                last = current_context[-1]["objects"][oid]
                pred_positions[oid] = [last["position"]["x"], last["position"]["y"]]

        step_mse = float(np.mean((pred_positions - gt_positions) ** 2))
        per_step_mse.append(step_mse)

        # Feed prediction back as context for next step
        pred_frame_dict = _frame_dict_from_prediction(
            pred_pos,
            current_context[-1],
            frame_num=context_frames + step_idx + 1,
        )
        current_context.append(pred_frame_dict)

        print(f"    Step {step_idx + 1}/{predict_frames}: MSE = {step_mse:.2f}"
              f"  (parsed {len(pred_pos)}/{num_objects} objects)")

    avg_mse = float(np.mean(per_step_mse))
    return {
        "scenario_type": scenario_type,
        "difficulty": difficulty,
        "seed": seed,
        "num_objects": num_objects,
        "per_step_mse": per_step_mse,
        "avg_mse": avg_mse,
        "success": True,
    }


def plot_comparison(results, baseline_mse=51.33, output_path="llm_vs_baseline_comparison.png"):
    """Generate comparison visualization."""
    valid = [r for r in results if r["success"]]
    if not valid:
        print("No valid results to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Bar chart of per-scenario average MSE ---
    ax = axes[0]
    names = [r["scenario_type"] for r in valid]
    mses = [r["avg_mse"] for r in valid]
    x = np.arange(len(names))
    bars = ax.bar(x, mses, color="steelblue", label="LLM", width=0.4)
    ax.axhline(y=baseline_mse, color="red", linestyle="--", linewidth=2, label=f"Baseline ({baseline_mse})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title("LLM MSE vs Naive Baseline (per scenario)")
    ax.legend()
    # Add value labels
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{mse:.1f}", ha="center", va="bottom", fontsize=9)

    # --- Right: Per-step MSE across prediction horizon ---
    ax = axes[1]
    for r in valid:
        steps = list(range(1, len(r["per_step_mse"]) + 1))
        ax.plot(steps, r["per_step_mse"], marker="o", label=r["scenario_type"])
    ax.axhline(y=baseline_mse, color="red", linestyle="--", linewidth=2, label=f"Baseline ({baseline_mse})")
    ax.set_xlabel("Prediction Step")
    ax.set_ylabel("MSE")
    ax.set_title("Error Accumulation Over Prediction Horizon")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def main():
    print("\nPhysicsLLMEngine LLM MSE Evaluation")
    print("=" * 80)

    # Load model
    model, tokenizer = load_model()

    # Held-out scenarios
    held_out_scenarios = [
        ("pong", 3),
        ("bowling", 3),
        ("ramp_roll", 3),
        ("angry_birds", 2),
        ("hourglass", 2),
        ("newtons_cradle", 3),
    ]

    print("\n" + "=" * 80)
    print("Evaluating on Held-Out Scenarios (10 context frames, 5 predicted)")
    print("=" * 80 + "\n")

    results = []
    rng = np.random.RandomState(2024)  # reproducible seeds

    for scenario_type, difficulty in held_out_scenarios:
        seed = int(rng.randint(10_000_000, 99_999_999))
        print(f"--- {scenario_type} (difficulty {difficulty}, seed {seed}) ---")

        try:
            result = evaluate_scenario(
                model=model,
                tokenizer=tokenizer,
                scenario_type=scenario_type,
                difficulty=difficulty,
                seed=seed,
                context_frames=10,
                predict_frames=5,
            )
            results.append(result)
            print(f"  => Average MSE: {result['avg_mse']:.2f}  ({result['num_objects']} objects)\n")
        except Exception as e:
            print(f"  => FAILED: {e}\n")
            results.append({
                "scenario_type": scenario_type,
                "difficulty": difficulty,
                "seed": seed,
                "error": str(e),
                "success": False,
            })

    # Save results
    output_file = Path(__file__).parent / "llm_mse_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_path = Path(__file__).parent / "llm_vs_baseline_comparison.png"
    plot_comparison(results, output_path=str(plot_path))

    # Summary
    valid_results = [r for r in results if r["success"]]
    baseline_mse = 51.33

    if valid_results:
        avg_mse = float(np.mean([r["avg_mse"] for r in valid_results]))

        print("=" * 80)
        print("LLM MSE RESULTS")
        print("=" * 80)
        print()
        print(f"Scenarios tested: {len(valid_results)}/{len(results)}")
        print(f"Average MSE:      {avg_mse:.2f}")
        print()
        print("Per-scenario breakdown:")
        for r in valid_results:
            steps_str = ", ".join(f"{m:.1f}" for m in r["per_step_mse"])
            print(f"  {r['scenario_type']:20s}  avg={r['avg_mse']:8.2f}  steps=[{steps_str}]")
        print()
        print("Baseline comparison:")
        print(f"  Naive baseline: {baseline_mse:.2f} MSE")
        print(f"  LLM model:      {avg_mse:.2f} MSE")

        if avg_mse < baseline_mse:
            improvement = (baseline_mse - avg_mse) / baseline_mse * 100
            print(f"  LLM beats baseline by {improvement:.1f}%!")
        else:
            excess = (avg_mse - baseline_mse) / baseline_mse * 100
            print(f"  LLM is {excess:.1f}% above baseline")
        print()

    print(f"Results saved to: {output_file}")
    print(f"Plot saved to:    {plot_path}")
    print("=" * 80)

    # Return avg_mse for scripting
    if valid_results:
        return float(np.mean([r["avg_mse"] for r in valid_results]))
    return None


if __name__ == "__main__":
    avg = main()
