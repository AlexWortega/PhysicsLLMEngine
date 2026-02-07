#!/usr/bin/env python3
"""
Render GT vs Predicted comparison GIFs for held-out scenarios.

Loads LFM2 + LoRA, runs autoregressive rollout on validation scenes,
renders side-by-side Ground Truth | Model Prediction animations.
"""
import re
import sys
import json
import os
import math
from pathlib import Path

sys.path.insert(0, "/home/alexw")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
import numpy as np

from src.training.data_loader import load_physics_scene, _format_header_text, _format_frame_text

# --- Config ---
CHECKPOINT_DIR = "/home/alexw/checkpoints/lfm2-scenarios/stage3/checkpoint-1000"
VAL_DIR = "/home/alexw/data_scenarios/val"
OUT_DIR = "/home/alexw/evaluation_results/comparison_gifs"

UNSEEN_TYPES = ["pong", "bowling", "ramp_roll", "angry_birds", "hourglass", "newtons_cradle",
                "planetary_rotation"]
SEEN_TYPES = ["head_on", "tower", "pendulum", "projectile", "billiards", "dominos"]

CONTEXT_FRAMES = 10
ROLLOUT_FRAMES = 20       # predict 20 frames autoregressively
MAX_OBJECTS_RENDER = 20   # cap rendering for very dense scenes
FPS = 15

OBJ_PATTERN = re.compile(
    r"obj_(\d+):\s*pos=\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)"
)


def load_model():
    import torch
    from unsloth import FastLanguageModel
    from peft import PeftModel

    print("Loading base LFM2-350M...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="LiquidAI/LFM2-350M",
        max_seq_length=8192,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    print(f"Loading LoRA from {CHECKPOINT_DIR}...", flush=True)
    model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model on {next(model.parameters()).device}", flush=True)
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=500):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=7000)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.1, do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_positions(text, num_objects):
    """Extract obj positions from predicted text. Returns dict {obj_id: (x, y)}."""
    positions = {}
    for m in OBJ_PATTERN.finditer(text):
        oid = int(m.group(1))
        if oid < num_objects:
            positions[oid] = (float(m.group(2)), float(m.group(3)))
    return positions


def get_scene_file(scenario_type, index=0):
    type_dir = Path(VAL_DIR) / scenario_type
    if not type_dir.exists():
        return None
    files = []
    for seed_dir in sorted(type_dir.iterdir()):
        if seed_dir.is_dir():
            for jf in sorted(seed_dir.glob("*.jsonl")):
                files.append(str(jf))
                if len(files) > index:
                    return files[index]
    return files[index] if len(files) > index else None


def run_rollout(model, tokenizer, header, frames, n_context, n_rollout):
    """Run autoregressive rollout. Returns list of dicts with parsed positions."""
    header_text = _format_header_text(header)
    context = header_text
    for i in range(min(n_context, len(frames))):
        context += _format_frame_text(frames[i])

    num_objects = header["object_count"]
    predicted_frames = []

    for step in range(n_rollout):
        target_idx = n_context + step
        if target_idx >= len(frames):
            break

        prompt = context + "Predict next frame:"
        pred_text = generate_text(model, tokenizer, prompt, max_new_tokens=400)
        positions = parse_positions(pred_text, num_objects)
        predicted_frames.append({"text": pred_text, "positions": positions, "frame_idx": target_idx})
        if (step + 1) % 5 == 0 or step == 0:
            print(f"    step {step+1}/{n_rollout} ({len(positions)} objs parsed)", flush=True)

        # Autoregressive: feed prediction back (not GT)
        # Build a synthetic frame text from prediction
        pred_frame_lines = [f"Frame {target_idx + 1}: predicted"]
        for oid in sorted(positions.keys()):
            px, py = positions[oid]
            pred_frame_lines.append(f"  obj_{oid}: pos=({px:.4f}, {py:.4f}), vel=(0.0000, 0.0000)")
        pred_frame_lines.append("")
        context += "\n".join(pred_frame_lines) + "\n"

        # Truncate context if too long
        if len(context) > 40000:
            header_end = context.find("Frame ")
            if header_end > 0:
                hdr = context[:header_end]
                rest = context[max(header_end, len(context) - 25000):]
                frame_start = rest.find("Frame ")
                if frame_start >= 0:
                    rest = rest[frame_start:]
                context = hdr + rest

    return predicted_frames


def extract_gt_positions(frame, num_objects):
    """Get positions from GT frame dict."""
    pos = {}
    for obj in frame.get("objects", []):
        oid = obj["id"]
        if oid < num_objects:
            pos[oid] = (obj["position"]["x"], obj["position"]["y"])
    return pos


def extract_object_info(header):
    """Get shape/size info from header for rendering."""
    infos = []
    for obj in header.get("objects", []):
        info = {
            "id": obj["id"],
            "type": obj.get("type", "circle"),
        }
        if info["type"] == "circle":
            info["radius"] = obj.get("radius", obj.get("size", {}).get("radius", 10))
        else:
            info["width"] = obj.get("size", {}).get("width", obj.get("width", 30))
            info["height"] = obj.get("size", {}).get("height", obj.get("height", 20))
        infos.append(info)
    return infos


def render_comparison_gif(scenario_type, header, frames, predicted_frames, output_path):
    """Render side-by-side GT vs Predicted GIF."""
    num_objects = min(header["object_count"], MAX_OBJECTS_RENDER)
    obj_infos = extract_object_info(header)[:num_objects]

    n_frames_render = len(predicted_frames)
    if n_frames_render == 0:
        return

    # Collect GT positions for the same frame range
    gt_all = []
    pred_all = []
    for pf in predicted_frames:
        idx = pf["frame_idx"]
        if idx < len(frames):
            gt_all.append(extract_gt_positions(frames[idx], num_objects))
        else:
            gt_all.append({})
        pred_all.append(pf["positions"])

    # Also include context frames as "warmup" in GT panel
    warmup = min(CONTEXT_FRAMES, len(frames))

    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    fig.patch.set_facecolor("#0f0f23")

    static_geom = header.get("static_geometry", [])

    for ax, title, color in [(ax_gt, "Ground Truth", "#2ecc71"), (ax_pred, "Prediction", "#e74c3c")]:
        ax.set_xlim(-10, 810)
        ax.set_ylim(-10, 610)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color=color, fontsize=13, fontweight="bold", pad=6)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Draw static geometry
        for sg in static_geom:
            if sg["type"] == "segment":
                p1, p2 = sg["p1"], sg["p2"]
                ax.plot([p1["x"], p2["x"]], [p1["y"], p2["y"]],
                        color="#555555", linewidth=2, zorder=1)
            elif sg["type"] == "circle":
                c = sg["center"]
                circ = plt.Circle((c["x"], c["y"]), sg["radius"],
                                  color="#555555", zorder=1)
                ax.add_patch(circ)

    fig.suptitle(
        f"{scenario_type.replace('_', ' ').title()}  |  seed={header.get('seed','?')}  "
        f"diff={header.get('difficulty','?')}  objs={header['object_count']}",
        color="#aaaaaa", fontsize=10, y=0.02
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Create object patches for both panels
    gt_patches = []
    pred_patches = []
    gt_color = "#3498db"
    pred_color = "#e67e22"

    for oi in obj_infos:
        if oi["type"] == "circle":
            r = oi.get("radius", 10)
            gc = plt.Circle((0, 0), r, color=gt_color, alpha=0.85, zorder=3)
            pc = plt.Circle((0, 0), r, color=pred_color, alpha=0.85, zorder=3)
            ax_gt.add_patch(gc)
            ax_pred.add_patch(pc)
            gt_patches.append(("circle", gc, r))
            pred_patches.append(("circle", pc, r))
        else:
            w = oi.get("width", 30)
            h = oi.get("height", 20)
            gr = patches.Rectangle((0, 0), w, h, color=gt_color, alpha=0.85, zorder=3)
            pr = patches.Rectangle((0, 0), w, h, color=pred_color, alpha=0.85, zorder=3)
            ax_gt.add_patch(gr)
            ax_pred.add_patch(pr)
            gt_patches.append(("rect", gr, w, h))
            pred_patches.append(("rect", pr, w, h))

    # Trail dots for trajectory visualization
    gt_trails = {oid: ax_gt.plot([], [], '.', color="#2ecc71", markersize=2, alpha=0.4, zorder=2)[0]
                 for oid in range(min(5, num_objects))}
    pred_trails = {oid: ax_pred.plot([], [], '.', color="#e74c3c", markersize=2, alpha=0.4, zorder=2)[0]
                   for oid in range(min(5, num_objects))}
    gt_trail_data = {oid: ([], []) for oid in range(min(5, num_objects))}
    pred_trail_data = {oid: ([], []) for oid in range(min(5, num_objects))}

    frame_label_gt = ax_gt.text(10, 585, "", color="#aaa", fontsize=8, zorder=10)
    frame_label_pred = ax_pred.text(10, 585, "", color="#aaa", fontsize=8, zorder=10)

    # Warmup frames (context) — show both panels identical
    warmup_positions = []
    for i in range(warmup):
        if i < len(frames):
            warmup_positions.append(extract_gt_positions(frames[i], num_objects))
        else:
            warmup_positions.append({})

    total_anim_frames = warmup + n_frames_render

    def update(fidx):
        if fidx < warmup:
            # Context phase: both panels show GT
            pos_gt = warmup_positions[fidx]
            pos_pred = warmup_positions[fidx]
            phase = "context"
            real_frame = fidx + 1
        else:
            # Rollout phase
            ri = fidx - warmup
            pos_gt = gt_all[ri] if ri < len(gt_all) else {}
            pos_pred = pred_all[ri] if ri < len(pred_all) else {}
            phase = "rollout"
            real_frame = CONTEXT_FRAMES + ri + 1

        def set_positions(patch_list, positions):
            for i, p in enumerate(patch_list):
                if i in positions:
                    px, py = positions[i]
                    if p[0] == "circle":
                        p[1].center = (px, py)
                    else:
                        _, rect, w, h = p
                        rect.set_xy((px - w/2, py - h/2))
                else:
                    # Hide off-screen
                    if p[0] == "circle":
                        p[1].center = (-100, -100)
                    else:
                        p[1].set_xy((-200, -200))

        set_positions(gt_patches, pos_gt)
        set_positions(pred_patches, pos_pred)

        # Update trails (first 5 objects)
        for oid in range(min(5, num_objects)):
            if oid in pos_gt:
                gt_trail_data[oid][0].append(pos_gt[oid][0])
                gt_trail_data[oid][1].append(pos_gt[oid][1])
                gt_trails[oid].set_data(gt_trail_data[oid][0], gt_trail_data[oid][1])
            if oid in pos_pred:
                pred_trail_data[oid][0].append(pos_pred[oid][0])
                pred_trail_data[oid][1].append(pos_pred[oid][1])
                pred_trails[oid].set_data(pred_trail_data[oid][0], pred_trail_data[oid][1])

        label = f"f:{real_frame}"
        if phase == "context":
            label += " [context]"
        frame_label_gt.set_text(label)
        frame_label_pred.set_text(label)
        return []

    anim = FuncAnimation(fig, update, frames=total_anim_frames, blit=False, interval=1000 // FPS)
    anim.save(output_path, writer=PillowWriter(fps=FPS))
    plt.close(fig)


def make_comparison_gallery(gif_dir, output_path, cols=3):
    """Create gallery image from comparison GIFs."""
    gifs = sorted([f for f in os.listdir(gif_dir) if f.startswith("cmp_") and f.endswith(".gif")])
    if not gifs:
        return

    rows = math.ceil(len(gifs) / cols)
    images = []
    for g in gifs:
        img = Image.open(os.path.join(gif_dir, g))
        # Get a frame in the rollout phase
        try:
            img.seek(min(25, img.n_frames - 1))
        except EOFError:
            img.seek(0)
        images.append(img.copy())

    w, h = images[0].size
    gallery = Image.new("RGB", (cols * w, rows * h), (15, 15, 35))
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        gallery.paste(img.convert("RGB"), (c * w, r * h))
    gallery.save(output_path)
    print(f"Comparison gallery: {output_path} ({cols}x{rows})", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, tokenizer = load_model()

    all_types = UNSEEN_TYPES + SEEN_TYPES
    total = len(all_types)

    for i, stype in enumerate(all_types):
        is_unseen = stype in UNSEEN_TYPES
        label = "UNSEEN" if is_unseen else "SEEN"
        print(f"\n[{i+1}/{total}] [{label}] {stype}", flush=True)

        sf = get_scene_file(stype, index=0)
        if sf is None:
            print(f"  No scenes found, skipping", flush=True)
            continue

        header, frames = load_physics_scene(sf)
        # Use fewer rollout frames for very large scenes
        n_roll = ROLLOUT_FRAMES if header["object_count"] <= 15 else 10
        print(f"  seed={header.get('seed','?')} objs={header['object_count']} "
              f"diff={header.get('difficulty','?')} rollout={n_roll}", flush=True)

        print(f"  Running autoregressive rollout...", flush=True)
        predicted = run_rollout(model, tokenizer, header, frames, CONTEXT_FRAMES, n_roll)
        print(f"  Got {len(predicted)} predicted frames", flush=True)

        gif_path = os.path.join(OUT_DIR, f"cmp_{label.lower()}_{stype}.gif")
        print(f"  Rendering comparison GIF...", flush=True)
        render_comparison_gif(stype, header, frames, predicted, gif_path)
        print(f"  Saved: {gif_path}", flush=True)

    # Second scene for 3 key unseen types (different seed)
    print("\n=== Second scenes for key unseen types ===", flush=True)
    for stype in ["pong", "bowling", "newtons_cradle"]:
        sf = get_scene_file(stype, index=1)
        if sf is None:
            continue
        header, frames = load_physics_scene(sf)
        n_roll = ROLLOUT_FRAMES if header["object_count"] <= 15 else 15
        print(f"\n  [{stype}] seed={header.get('seed','?')} objs={header['object_count']} "
              f"diff={header.get('difficulty','?')}", flush=True)
        predicted = run_rollout(model, tokenizer, header, frames, CONTEXT_FRAMES, n_roll)
        print(f"  {len(predicted)} frames predicted", flush=True)
        gif_path = os.path.join(OUT_DIR, f"cmp_unseen_{stype}_v2.gif")
        render_comparison_gif(stype, header, frames, predicted, gif_path)
        print(f"  Saved: {gif_path}", flush=True)

    # Gallery
    gallery_path = os.path.join(OUT_DIR, "comparison_gallery.png")
    make_comparison_gallery(OUT_DIR, gallery_path, cols=3)

    print(f"\nAll done! Files in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
