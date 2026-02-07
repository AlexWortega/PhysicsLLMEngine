#!/usr/bin/env python3
"""
Checkpoint evaluator for LFM2-350M fine-tuning.

Watches for new stage checkpoints, evaluates on val scenarios,
plots per-scenario metrics, and logs everything to W&B.

Usage:
    # Run as watcher (loops until all stages done)
    python scripts/eval_checkpoints.py

    # Evaluate a specific checkpoint
    python scripts/eval_checkpoints.py --checkpoint checkpoints/lfm2-scenarios/stage0

    # Quick test
    python scripts/eval_checkpoints.py --scenes-per-type 5 --max-context 3
"""

import argparse
import json
import os
import sys
import time
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


# Scenario classification
UNSEEN_SCENARIOS = {"pong", "bowling", "ramp_roll", "angry_birds", "hourglass", "newtons_cradle"}
SEEN_SCENARIOS = {
    "avalanche", "basketball", "billiards", "breakout", "bridge",
    "chain", "conveyor", "dominos", "explosion", "funnel",
    "head_on", "jenga", "marble_run", "orbit", "pendulum",
    "pinball", "plinko", "projectile", "pyramid", "seesaw",
    "ski_jump", "tower", "wind", "wrecking_ball",
}
CATEGORIES = {
    "collision": ["billiards", "breakout", "explosion", "head_on", "projectile", "bowling"],
    "stacking": ["bridge", "dominos", "jenga", "pyramid", "tower"],
    "ramp": ["funnel", "marble_run", "plinko", "ski_jump", "ramp_roll"],
    "constraint": ["chain", "pendulum", "seesaw", "wrecking_ball", "newtons_cradle"],
    "minigame": ["basketball", "pinball", "angry_birds", "pong"],
    "complex": ["avalanche", "conveyor", "hourglass", "orbit", "wind"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LFM2 checkpoints on val scenarios")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/lfm2-scenarios",
                        help="Directory containing stage checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Evaluate a specific checkpoint (skip watching)")
    parser.add_argument("--val-dir", type=str, default="data_scenarios/val",
                        help="Validation data directory")
    parser.add_argument("--output-dir", type=str, default="evaluation_results/lfm2-scenarios",
                        help="Output directory for results")
    parser.add_argument("--scenes-per-type", type=int, default=20,
                        help="Number of val scenes to evaluate per scenario type")
    parser.add_argument("--max-context", type=int, default=5,
                        help="Context frames for evaluation examples")
    parser.add_argument("--max-total-tokens", type=int, default=8192,
                        help="Token budget per example")
    parser.add_argument("--wandb-project", type=str, default="physics-llm")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=300,
                        help="Seconds between checkpoint checks (watcher mode)")
    parser.add_argument("--total-stages", type=int, default=5,
                        help="Total expected training stages")
    return parser.parse_args()


def extract_predictions(text: str) -> List[Dict[str, Any]]:
    """Parse model output text into object predictions."""
    objects = []
    _num = r'([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)'
    pattern = re.compile(
        r'obj_(\d+):\s*pos=\s*\(' + _num + r',\s*' + _num + r'\),\s*vel=\s*\(' + _num + r',\s*' + _num + r'\)'
        r'(?:,\s*a=\s*' + _num + r',\s*av=\s*' + _num + r')?'
    )
    for m in pattern.finditer(text):
        try:
            obj = {
                "id": int(m.group(1)),
                "px": float(m.group(2)), "py": float(m.group(3)),
                "vx": float(m.group(4)), "vy": float(m.group(5)),
            }
            if m.group(6):
                obj["angle"] = float(m.group(6))
                obj["ang_vel"] = float(m.group(7))
            objects.append(obj)
        except (ValueError, TypeError):
            continue
    return objects


def compute_frame_metrics(pred_objects, true_objects) -> Dict[str, float]:
    """Compute position and velocity MSE between predicted and true frames."""
    if not pred_objects or not true_objects:
        return {"pos_mse": float("inf"), "vel_mse": float("inf"), "parse_fail": 1.0}

    # Match by object ID
    true_map = {o["id"]: o for o in true_objects}
    pos_errors, vel_errors = [], []

    for pred in pred_objects:
        oid = pred["id"]
        if oid not in true_map:
            continue
        true = true_map[oid]
        pos_errors.append((pred["px"] - true["px"])**2 + (pred["py"] - true["py"])**2)
        vel_errors.append((pred["vx"] - true["vx"])**2 + (pred["vy"] - true["vy"])**2)

    if not pos_errors:
        return {"pos_mse": float("inf"), "vel_mse": float("inf"), "parse_fail": 1.0}

    return {
        "pos_mse": float(np.mean(pos_errors)),
        "vel_mse": float(np.mean(vel_errors)),
        "parse_fail": 0.0,
    }


def load_model_and_tokenizer(checkpoint_path: str):
    """Load LFM2-350M with LoRA adapter from checkpoint."""
    from unsloth import FastLanguageModel

    adapter_path = Path(checkpoint_path) / "adapter"
    if not adapter_path.exists():
        # Try direct path
        adapter_path = Path(checkpoint_path)

    print(f"Loading base model LFM2-350M...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="LiquidAI/LFM2-350M",
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=False,
    )

    print(f"Loading LoRA adapter from {adapter_path}...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_prediction(model, tokenizer, input_text: str, max_new_tokens: int = 512) -> str:
    """Generate model prediction for given input."""
    import torch

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def compute_val_loss(model, tokenizer, examples: List[Dict]) -> float:
    """Compute average loss on examples."""
    import torch

    total_loss = 0.0
    count = 0

    # Detect model dtype for autocast (Unsloth loads in bf16)
    model_dtype = next(model.parameters()).dtype

    for ex in examples:
        text = ex["input"] + "\n\n" + ex["output"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=model_dtype):
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            count += 1

    return total_loss / max(count, 1)


def evaluate_checkpoint(
    model, tokenizer, val_dir: str, scenes_per_type: int,
    max_context: int, max_total_tokens: int,
) -> Dict[str, Any]:
    """Evaluate model on val set, returning per-scenario metrics."""
    import glob
    import random
    from src.training.data_loader import load_physics_scene, format_training_example

    val_path = Path(val_dir)
    scenario_dirs = sorted([d for d in val_path.iterdir() if d.is_dir()])

    results = {}
    all_losses = []

    for sd in scenario_dirs:
        scenario_name = sd.name
        files = list(sd.rglob("*.jsonl"))
        if not files:
            continue

        # Sample scenes
        random.seed(42)
        sample_files = random.sample(files, min(scenes_per_type, len(files)))

        scenario_metrics = defaultdict(list)
        scenario_losses = []

        for scene_file in sample_files:
            try:
                header, frames = load_physics_scene(str(scene_file))
            except Exception:
                continue

            if len(frames) < max_context + 2:
                continue

            # Pick a target frame in the middle of the scene
            target_idx = min(max_context + 1, len(frames) - 1)
            ctx_frames = frames[:target_idx]
            target_frame = frames[target_idx]

            # Format with token budget
            input_text, output_text = format_training_example(
                header, ctx_frames, target_frame, max_total_tokens=max_total_tokens,
            )

            # Compute loss
            loss = compute_val_loss(model, tokenizer, [{"input": input_text, "output": output_text}])
            scenario_losses.append(loss)
            all_losses.append(loss)

            # Generate prediction and compute metrics
            pred_text = generate_prediction(model, tokenizer, input_text)
            pred_objects = extract_predictions(pred_text)

            # Parse ground truth
            true_objects = []
            for obj in target_frame["objects"]:
                true_objects.append({
                    "id": obj["id"],
                    "px": obj["position"]["x"], "py": obj["position"]["y"],
                    "vx": obj["velocity"]["x"], "vy": obj["velocity"]["y"],
                })

            metrics = compute_frame_metrics(pred_objects, true_objects)
            for k, v in metrics.items():
                scenario_metrics[k].append(v)

        # Aggregate per-scenario
        is_unseen = scenario_name in UNSEEN_SCENARIOS
        category = next((cat for cat, types in CATEGORIES.items() if scenario_name in types), "unknown")

        results[scenario_name] = {
            "loss": float(np.mean(scenario_losses)) if scenario_losses else float("inf"),
            "pos_mse": float(np.mean(scenario_metrics["pos_mse"])) if scenario_metrics["pos_mse"] else float("inf"),
            "vel_mse": float(np.mean(scenario_metrics["vel_mse"])) if scenario_metrics["vel_mse"] else float("inf"),
            "parse_fail_rate": float(np.mean(scenario_metrics["parse_fail"])) if scenario_metrics["parse_fail"] else 1.0,
            "n_scenes": len(scenario_losses),
            "is_unseen": is_unseen,
            "category": category,
        }
        status = "UNSEEN" if is_unseen else "seen"
        print(f"  {scenario_name} ({status}): loss={results[scenario_name]['loss']:.4f}, "
              f"pos_mse={results[scenario_name]['pos_mse']:.2f}, "
              f"parse_fail={results[scenario_name]['parse_fail_rate']:.0%}")

    # Overall aggregates
    seen_losses = [r["loss"] for s, r in results.items() if not r["is_unseen"] and r["loss"] < float("inf")]
    unseen_losses = [r["loss"] for s, r in results.items() if r["is_unseen"] and r["loss"] < float("inf")]
    seen_pos = [r["pos_mse"] for s, r in results.items() if not r["is_unseen"] and r["pos_mse"] < float("inf")]
    unseen_pos = [r["pos_mse"] for s, r in results.items() if r["is_unseen"] and r["pos_mse"] < float("inf")]

    summary = {
        "overall_loss": float(np.mean(all_losses)) if all_losses else float("inf"),
        "seen_loss": float(np.mean(seen_losses)) if seen_losses else float("inf"),
        "unseen_loss": float(np.mean(unseen_losses)) if unseen_losses else float("inf"),
        "seen_pos_mse": float(np.mean(seen_pos)) if seen_pos else float("inf"),
        "unseen_pos_mse": float(np.mean(unseen_pos)) if unseen_pos else float("inf"),
        "generalization_gap": (float(np.mean(unseen_losses)) - float(np.mean(seen_losses))) if seen_losses and unseen_losses else float("inf"),
        "per_scenario": results,
    }

    # Per-category aggregates
    for cat, types in CATEGORIES.items():
        cat_losses = [r["loss"] for s, r in results.items() if s in types and r["loss"] < float("inf")]
        cat_pos = [r["pos_mse"] for s, r in results.items() if s in types and r["pos_mse"] < float("inf")]
        summary[f"cat_{cat}_loss"] = float(np.mean(cat_losses)) if cat_losses else float("inf")
        summary[f"cat_{cat}_pos_mse"] = float(np.mean(cat_pos)) if cat_pos else float("inf")

    return summary


def plot_results(results: Dict, stage: int, output_dir: str) -> List[str]:
    """Create evaluation plots and return paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots = []

    per_scenario = results["per_scenario"]

    # 1. Per-scenario loss bar chart
    fig, ax = plt.subplots(figsize=(16, 6))
    scenarios = sorted(per_scenario.keys())
    losses = [per_scenario[s]["loss"] for s in scenarios]
    colors = ["#e74c3c" if per_scenario[s]["is_unseen"] else "#3498db" for s in scenarios]
    bars = ax.bar(range(len(scenarios)), losses, color=colors)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Loss")
    ax.set_title(f"Stage {stage}: Per-Scenario Val Loss (red=unseen)")
    ax.axhline(y=results["seen_loss"], color="#3498db", linestyle="--", alpha=0.5, label=f"seen avg={results['seen_loss']:.3f}")
    ax.axhline(y=results["unseen_loss"], color="#e74c3c", linestyle="--", alpha=0.5, label=f"unseen avg={results['unseen_loss']:.3f}")
    ax.legend()
    plt.tight_layout()
    path = str(out / f"stage{stage}_loss_by_scenario.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plots.append(path)

    # 2. Per-scenario position MSE
    fig, ax = plt.subplots(figsize=(16, 6))
    pos_mses = [min(per_scenario[s]["pos_mse"], 1e6) for s in scenarios]
    ax.bar(range(len(scenarios)), pos_mses, color=colors)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Position MSE")
    ax.set_title(f"Stage {stage}: Per-Scenario Position MSE (red=unseen)")
    ax.set_yscale("log")
    plt.tight_layout()
    path = str(out / f"stage{stage}_pos_mse_by_scenario.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plots.append(path)

    # 3. Category breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cats = list(CATEGORIES.keys())
    cat_losses = [results.get(f"cat_{c}_loss", 0) for c in cats]
    cat_pos = [results.get(f"cat_{c}_pos_mse", 0) for c in cats]

    axes[0].bar(cats, cat_losses, color="#2ecc71")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Stage {stage}: Loss by Category")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(cats, [min(p, 1e6) for p in cat_pos], color="#9b59b6")
    axes[1].set_ylabel("Position MSE")
    axes[1].set_title(f"Stage {stage}: Pos MSE by Category")
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = str(out / f"stage{stage}_category_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plots.append(path)

    # 4. Seen vs Unseen comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    x = ["Seen (24 types)", "Unseen (6 types)"]
    y_loss = [results["seen_loss"], results["unseen_loss"]]
    bars = ax.bar(x, y_loss, color=["#3498db", "#e74c3c"])
    ax.set_ylabel("Loss")
    ax.set_title(f"Stage {stage}: Generalization Gap = {results['generalization_gap']:.4f}")
    for bar, val in zip(bars, y_loss):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.4f}",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = str(out / f"stage{stage}_seen_vs_unseen.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plots.append(path)

    return plots


def log_to_wandb(results: Dict, stage: int, plots: List[str], run):
    """Log evaluation results to W&B."""
    import wandb

    # Log scalar metrics
    metrics = {
        f"eval/overall_loss": results["overall_loss"],
        f"eval/seen_loss": results["seen_loss"],
        f"eval/unseen_loss": results["unseen_loss"],
        f"eval/seen_pos_mse": results["seen_pos_mse"],
        f"eval/unseen_pos_mse": results["unseen_pos_mse"],
        f"eval/generalization_gap": results["generalization_gap"],
        f"eval/stage": stage,
    }

    # Per-category
    for cat in CATEGORIES:
        metrics[f"eval/cat_{cat}_loss"] = results.get(f"cat_{cat}_loss", float("inf"))
        metrics[f"eval/cat_{cat}_pos_mse"] = results.get(f"cat_{cat}_pos_mse", float("inf"))

    # Per-scenario
    for scenario, data in results["per_scenario"].items():
        metrics[f"eval/scenario/{scenario}/loss"] = data["loss"]
        metrics[f"eval/scenario/{scenario}/pos_mse"] = data["pos_mse"]
        metrics[f"eval/scenario/{scenario}/parse_fail"] = data["parse_fail_rate"]

    wandb.log(metrics, step=stage)

    # Log plots as images
    for plot_path in plots:
        name = Path(plot_path).stem
        wandb.log({f"eval/{name}": wandb.Image(plot_path)}, step=stage)

    # Log results table
    table_data = []
    for scenario, data in sorted(results["per_scenario"].items()):
        table_data.append([
            scenario, data["category"],
            "unseen" if data["is_unseen"] else "seen",
            data["loss"], data["pos_mse"], data["vel_mse"],
            data["parse_fail_rate"], data["n_scenes"],
        ])

    table = wandb.Table(
        columns=["scenario", "category", "split", "loss", "pos_mse", "vel_mse", "parse_fail", "n_scenes"],
        data=table_data,
    )
    wandb.log({f"eval/scenario_table_stage{stage}": table}, step=stage)


def get_evaluated_stages(output_dir: str) -> set:
    """Get set of already-evaluated stages."""
    out = Path(output_dir)
    evaluated = set()
    for f in out.glob("stage*_results.json"):
        try:
            stage = int(f.stem.split("_")[0].replace("stage", ""))
            evaluated.add(stage)
        except ValueError:
            continue
    return evaluated


def get_available_checkpoints(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """Find completed stage checkpoints (have metadata.json)."""
    ckpt_path = Path(checkpoint_dir)
    checkpoints = []
    for d in ckpt_path.glob("stage*"):
        if (d / "metadata.json").exists() and (d / "adapter").exists():
            try:
                stage = int(d.name.replace("stage", ""))
                checkpoints.append((stage, str(d)))
            except ValueError:
                continue
    return sorted(checkpoints)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup W&B
    import wandb
    mode = "offline" if args.wandb_offline else None
    run = wandb.init(
        project=args.wandb_project,
        name="lfm2-scenario-eval",
        config=vars(args),
        mode=mode,
        job_type="evaluation",
    )

    if args.checkpoint:
        # Single checkpoint evaluation
        print(f"Evaluating checkpoint: {args.checkpoint}")
        model, tokenizer = load_model_and_tokenizer(args.checkpoint)

        stage = 0
        # Try to extract stage number
        m = re.search(r'stage(\d+)', args.checkpoint)
        if m:
            stage = int(m.group(1))

        results = evaluate_checkpoint(
            model, tokenizer, args.val_dir,
            args.scenes_per_type, args.max_context, args.max_total_tokens,
        )

        # Save, plot, log
        with open(output_dir / f"stage{stage}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        plots = plot_results(results, stage, str(output_dir))
        log_to_wandb(results, stage, plots, run)

        print(f"\nStage {stage} evaluation complete!")
        print(f"  Overall loss: {results['overall_loss']:.4f}")
        print(f"  Seen loss: {results['seen_loss']:.4f}")
        print(f"  Unseen loss: {results['unseen_loss']:.4f}")
        print(f"  Gap: {results['generalization_gap']:.4f}")

    else:
        # Watcher mode
        print(f"Watching {args.checkpoint_dir} for checkpoints...")
        print(f"Will evaluate on {args.val_dir} ({args.scenes_per_type} scenes/type)")
        print(f"Polling every {args.poll_interval}s")

        model = None
        tokenizer = None

        while True:
            evaluated = get_evaluated_stages(str(output_dir))
            available = get_available_checkpoints(args.checkpoint_dir)
            new_checkpoints = [(s, p) for s, p in available if s not in evaluated]

            if new_checkpoints:
                for stage, ckpt_path in new_checkpoints:
                    print(f"\n{'='*60}")
                    print(f"New checkpoint found: stage {stage}")
                    print(f"{'='*60}")

                    # Load model (reload each time for fresh adapter)
                    del model
                    import torch
                    torch.cuda.empty_cache()
                    model, tokenizer = load_model_and_tokenizer(ckpt_path)

                    results = evaluate_checkpoint(
                        model, tokenizer, args.val_dir,
                        args.scenes_per_type, args.max_context, args.max_total_tokens,
                    )

                    # Save results
                    with open(output_dir / f"stage{stage}_results.json", "w") as f:
                        json.dump(results, f, indent=2)

                    # Plot
                    plots = plot_results(results, stage, str(output_dir))

                    # Log to W&B
                    log_to_wandb(results, stage, plots, run)

                    print(f"\nStage {stage} evaluation complete!")
                    print(f"  Overall loss: {results['overall_loss']:.4f}")
                    print(f"  Seen: {results['seen_loss']:.4f} | Unseen: {results['unseen_loss']:.4f}")
                    print(f"  Gap: {results['generalization_gap']:.4f}")

            # Check if all stages evaluated
            if len(evaluated | {s for s, _ in new_checkpoints}) >= args.total_stages:
                print(f"\nAll {args.total_stages} stages evaluated. Generating summary...")
                _generate_summary(output_dir, run)
                break

            # Wait and poll
            print(f"[{time.strftime('%H:%M:%S')}] Waiting for new checkpoints... "
                  f"({len(evaluated | {s for s, _ in new_checkpoints})}/{args.total_stages} done)")
            time.sleep(args.poll_interval)

    wandb.finish()
    print("Done!")


def _generate_summary(output_dir: Path, run):
    """Generate cross-stage summary plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    # Load all stage results
    stage_results = {}
    for f in sorted(output_dir.glob("stage*_results.json")):
        stage = int(f.stem.split("_")[0].replace("stage", ""))
        with open(f) as fh:
            stage_results[stage] = json.load(fh)

    if len(stage_results) < 2:
        return

    stages = sorted(stage_results.keys())

    # Training progress: loss over stages
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    seen_losses = [stage_results[s]["seen_loss"] for s in stages]
    unseen_losses = [stage_results[s]["unseen_loss"] for s in stages]
    overall_losses = [stage_results[s]["overall_loss"] for s in stages]

    axes[0].plot(stages, seen_losses, "b-o", label="Seen (24 types)")
    axes[0].plot(stages, unseen_losses, "r-o", label="Unseen (6 types)")
    axes[0].plot(stages, overall_losses, "g--", label="Overall", alpha=0.5)
    axes[0].set_xlabel("Stage")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Val Loss over Training Stages")
    axes[0].legend()

    # Generalization gap
    gaps = [stage_results[s]["generalization_gap"] for s in stages]
    axes[1].plot(stages, gaps, "m-o")
    axes[1].set_xlabel("Stage")
    axes[1].set_ylabel("Gap (unseen - seen)")
    axes[1].set_title("Generalization Gap over Stages")
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Category losses at final stage
    final = stage_results[stages[-1]]
    cats = list(CATEGORIES.keys())
    cat_losses = [final.get(f"cat_{c}_loss", 0) for c in cats]
    axes[2].bar(cats, cat_losses, color=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"])
    axes[2].set_ylabel("Loss")
    axes[2].set_title(f"Final Stage: Loss by Category")
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = str(output_dir / "training_progress_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    wandb.log({"eval/training_progress": wandb.Image(path)})
    print(f"Summary plot saved: {path}")


if __name__ == "__main__":
    main()
