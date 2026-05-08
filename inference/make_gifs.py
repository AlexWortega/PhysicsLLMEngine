"""
Run all 30 demo scenarios × 200 frames through lfm2-scenarios-Q4_K_M.gguf
and render each rollout as an animated GIF.

Usage:
    python3 make_gifs.py [--out /path/to/gifs] [--frames 200] [--scenarios scen1,scen2]
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from PIL import Image
import io

# ── paths ──────────────────────────────────────────────────────────────────
GGUF = "/home/alexw/physics-llm-debug/lfm2-scenarios-Q4_K_M.gguf"
EXAMPLES = Path("/home/alexw/Projects/physics-llm-space/backend/examples")

# ── prompt formatting (mirrors TS promptFormat.ts) ─────────────────────────
def fmt(n: float, d: int = 4) -> str:
    return f"{n:.{d}f}"

def fmt_header(h: dict) -> str:
    lines = [f"Scene: {h.get('description', '')}"]
    g = h.get("gravity", {"x": 0, "y": 0})
    lines.append(f"Gravity: ({g['x']}, {g['y']})")
    lines.append(f"Timestep: {h.get('timestep', 0.01667):.5f}")
    if h.get("scenario_type"):
        lines.append(f"Type: {h['scenario_type']}")
    if h.get("difficulty") is not None:
        lines.append(f"Difficulty: {h['difficulty']}")
    for sg in (h.get("static_geometry") or []):
        pass  # handled below
    stat = h.get("static_geometry") or []
    if stat:
        parts = []
        for sg in stat:
            if sg["type"] == "segment":
                p1, p2 = sg["p1"], sg["p2"]
                parts.append(f"seg ({round(p1['x'])},{round(p1['y'])})-({round(p2['x'])},{round(p2['y'])})")
            elif sg["type"] == "circle":
                c = sg["center"]
                parts.append(f"peg ({round(c['x'])},{round(c['y'])}) r={round(sg['radius'])}")
        if parts:
            lines.append(f"Static: {'; '.join(parts)}")
    cons = h.get("constraints") or []
    if cons:
        parts = [f"{c['type']} {c['body_a']}->{c['body_b']}" for c in cons]
        lines.append(f"Constraints: {'; '.join(parts)}")
    lines.append("")
    return "\n".join(lines)

def fmt_frame(fr: dict) -> str:
    lines = [f"Frame {fr['frame']}: {fr.get('description', '')}"]
    for o in fr["objects"]:
        p = o["position"]
        v = o.get("velocity", {"x": 0, "y": 0})
        a = o.get("angle", 0)
        av = o.get("angular_velocity", 0)
        s = f"  obj_{o['id']}: pos=({fmt(p['x'])}, {fmt(p['y'])}), vel=({fmt(v['x'])}, {fmt(v['y'])})"
        if abs(a) > 0.001 or abs(av) > 0.001:
            s += f", a={fmt(a)}, av={fmt(av)}"
        lines.append(s)
    lines.append("")
    return "\n".join(lines)

# ── object regex ────────────────────────────────────────────────────────────
OBJ_RE = re.compile(
    r"obj_(\d+):\s*pos=\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)"
    r"(?:,\s*vel=\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\))?"
    r"(?:,\s*a=(-?[\d.]+))?"
)

def parse_frame(text: str, n_obj: int) -> dict[int, dict]:
    """Return {id: {x,y,vx,vy,angle}} for all parsed objects."""
    objs = {}
    for m in OBJ_RE.finditer(text):
        oid = int(m.group(1))
        if oid >= n_obj:
            continue
        objs[oid] = {
            "x": float(m.group(2)),
            "y": float(m.group(3)),
            "vx": float(m.group(4)) if m.group(4) else 0.0,
            "vy": float(m.group(5)) if m.group(5) else 0.0,
            "angle": float(m.group(6)) if m.group(6) else 0.0,
        }
    return objs

# ── rollout ─────────────────────────────────────────────────────────────────
def rollout(
    llm,
    header: dict,
    initial_frames: list[dict],
    n_steps: int,
    verbose: bool = False,
) -> list[dict[int, dict]]:
    n_obj = header.get("object_count", len(initial_frames[0]["objects"]))
    # seed_frame: only last initial frame (1-frame prompt is 1.75x faster)
    seed = initial_frames[-1:]
    ctx = fmt_header(header) + "".join(fmt_frame(f) for f in seed)

    # Build initial state from the last seed frame
    prev: dict[int, dict] = {
        o["id"]: {
            "x": o["position"]["x"],
            "y": o["position"]["y"],
            "vx": o.get("velocity", {}).get("x", 0),
            "vy": o.get("velocity", {}).get("y", 0),
            "angle": o.get("angle", 0),
        }
        for o in initial_frames[-1]["objects"]
    }

    frames: list[dict[int, dict]] = [prev]
    last_idx = initial_frames[-1]["frame"]

    # Keep last N frames in ctx to stay within budget
    MAX_CTX_FRAMES = 4
    ctx_frames: list[str] = [fmt_frame(f) for f in seed]

    for step in range(n_steps):
        prompt = ctx + "Predict next frame:"
        out = llm(prompt, max_tokens=600, temperature=0.0, top_p=1.0,
                  stop=["Predict next frame:"])
        text: str = out["choices"][0]["text"]

        # Extract first frame block
        ms = list(re.finditer(r"Frame\s+\d+:", text))
        if ms:
            end = ms[1].start() if len(ms) > 1 else len(text)
            first = text[ms[0].start():end]
            had_header = True
        else:
            first = text
            had_header = False

        parsed = parse_frame(first, n_obj)

        if verbose:
            n_parsed = len(parsed)
            if not had_header:
                print(f"  step {step+1:3d} | DRIFT (no header) | parsed {n_parsed}/{n_obj}")
            else:
                dx = [abs(parsed[i]["x"] - prev[i]["x"]) for i in parsed if i in prev]
                mean_d = sum(dx) / len(dx) if dx else 0
                print(f"  step {step+1:3d} | parsed {n_parsed}/{n_obj} | mean Δx {mean_d:.3f}")

        # Fill missing objects from prev
        full: dict[int, dict] = dict(prev)
        full.update(parsed)
        prev = full
        frames.append(dict(prev))
        last_idx += 1

        # Build append text for context
        if had_header and first.strip():
            append = first.rstrip() + "\n"
        else:
            # Synthesize a frame header if model drifted
            obj_lines = "\n".join(
                f"  obj_{i}: pos=({v['x']:.4f}, {v['y']:.4f}), vel=({v['vx']:.4f}, {v['vy']:.4f})"
                for i, v in sorted(full.items())
            )
            append = f"Frame {last_idx}: All objects are in motion.\n{obj_lines}\n"

        ctx_frames.append(append)
        if len(ctx_frames) > MAX_CTX_FRAMES:
            ctx_frames = ctx_frames[-MAX_CTX_FRAMES:]

        ctx = fmt_header(header) + "".join(ctx_frames)

    return frames

# ── rendering ───────────────────────────────────────────────────────────────
W, H = 800, 600
COLORS = plt.cm.tab20.colors  # 20 distinct colours, cycle for more objects

def draw_frame(
    ax: plt.Axes,
    header: dict,
    frame_state: dict[int, dict],
    obj_shapes: dict[int, dict],
    title: str,
) -> None:
    ax.cla()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=8, pad=2)

    # Draw static geometry
    for sg in (header.get("static_geometry") or []):
        if sg["type"] == "segment":
            p1, p2 = sg["p1"], sg["p2"]
            ax.plot(
                [p1["x"], p2["x"]],
                [H - p1["y"], H - p2["y"]],
                color="#888888", lw=2, solid_capstyle="round",
            )
        elif sg["type"] == "circle":
            c = sg["center"]
            circle = plt.Circle(
                (c["x"], H - c["y"]), sg["radius"],
                color="#888888", fill=False, lw=1.5,
            )
            ax.add_patch(circle)

    # Draw objects
    for oid, state in sorted(frame_state.items()):
        color = COLORS[oid % len(COLORS)]
        shape = obj_shapes.get(oid, {"type": "circle", "radius": 10})
        x, y = state["x"], H - state["y"]  # flip Y
        angle_deg = -np.degrees(state.get("angle", 0))  # flip for screen coords

        if shape["type"] == "circle":
            r = shape.get("radius", 10)
            patch = plt.Circle((x, y), r, color=color, zorder=3)
            ax.add_patch(patch)
            # small angle indicator
            dx = r * np.cos(np.radians(angle_deg))
            dy = r * np.sin(np.radians(angle_deg))
            ax.plot([x, x + dx], [y, y + dy], color="white", lw=1, alpha=0.6, zorder=4)
        elif shape["type"] == "rectangle":
            w, h = shape.get("width", 20), shape.get("height", 10)
            # matplotlib Rectangle anchored at bottom-left, we want center
            transform = matplotlib.transforms.Affine2D().rotate_deg_around(x, y, angle_deg) + ax.transData
            rect = mpatches.Rectangle(
                (x - w / 2, y - h / 2), w, h,
                color=color, zorder=3, transform=transform,
            )
            ax.add_patch(rect)


def render_gif(
    header: dict,
    frames: list[dict[int, dict]],
    out_path: Path,
    fps: int = 20,
    skip: int = 3,
) -> None:
    """Render rollout as GIF. `skip` = take every Nth frame to keep file small."""
    # Build per-object shape metadata from header
    obj_shapes: dict[int, dict] = {}
    for o in header.get("objects", []):
        obj_shapes[o["id"]] = {
            "type": o.get("type", "circle"),
            "radius": o.get("radius", 10),
            "width": o.get("width", 20),
            "height": o.get("height", 10),
        }

    name = header.get("description", "")[:40]
    fig, ax = plt.subplots(1, 1, figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)

    pil_frames: list[Image.Image] = []
    selected = frames[::skip]
    for fi, state in enumerate(selected):
        draw_frame(ax, header, state, obj_shapes, f"{name}  [{fi*skip+1}/{len(frames)}]")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)

    if not pil_frames:
        return

    delay = max(20, int(1000 / fps))  # ms per frame
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=delay,
        loop=0,
        optimize=False,
    )


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/home/alexw/physics-llm-debug/gifs")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--skip", type=int, default=3, help="render every Nth predicted frame")
    parser.add_argument("--scenarios", default="", help="comma-sep list; empty = all")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_jsonl = sorted(EXAMPLES.glob("*.jsonl"))
    if args.scenarios:
        wanted = set(args.scenarios.split(","))
        all_jsonl = [p for p in all_jsonl if p.stem in wanted]

    print(f"Loading model from {GGUF} ...")
    from llama_cpp import Llama
    llm = Llama(
        model_path=GGUF,
        n_ctx=8192,
        n_threads=min(16, os.cpu_count() or 8),
        verbose=False,
    )
    print(f"Model loaded. Will run {len(all_jsonl)} scenarios × {args.frames} frames.\n")

    total_t0 = time.time()
    for sc_path in all_jsonl:
        name = sc_path.stem
        lines = [ln for ln in sc_path.read_text().splitlines() if ln.strip()]
        header = json.loads(lines[0])
        initial = [json.loads(ln) for ln in lines[1:5]]

        print(f"[{name}] {header.get('description','')[:60]}")
        t0 = time.time()
        frames = rollout(llm, header, initial, n_steps=args.frames, verbose=args.verbose)
        dt = time.time() - t0
        print(f"  rollout: {len(frames)} frames in {dt:.1f}s ({len(frames)/dt:.2f} fps)")

        gif_path = out_dir / f"{name}.gif"
        render_gif(header, frames, gif_path, fps=args.fps, skip=args.skip)
        sz = gif_path.stat().st_size // 1024
        print(f"  → {gif_path.name} ({sz} KB)\n")

    total_dt = time.time() - total_t0
    print(f"All done in {total_dt/60:.1f} min. GIFs in {out_dir}")
    print(f"\nSending via wormhole...")
    # Zip all GIFs and send
    import zipfile
    zip_path = out_dir.parent / "physics_gifs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for gif in sorted(out_dir.glob("*.gif")):
            zf.write(gif, gif.name)
    zip_sz = zip_path.stat().st_size // 1024 // 1024
    print(f"Zipped {zip_sz} MB → {zip_path}")
    os.execlp("wormhole", "wormhole", "send", str(zip_path))


if __name__ == "__main__":
    main()
