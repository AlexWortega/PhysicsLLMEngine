"""Local repro of what the browser code does.

Loads one of the bundled scenarios, builds the prompt with the SAME formatting
as our TS port, and runs llama-cpp generation. Then prints the raw output.
"""
from __future__ import annotations
import json, sys, re
from pathlib import Path

# --- match TS promptFormat.ts ---

def fmt(n: float, d: int) -> str:
    return f"{n:.{d}f}"

def fmt_header(h: dict) -> str:
    lines = [f"Scene: {h.get('description','')}"]
    g = h.get("gravity", {"x": 0, "y": 0})
    lines.append(f"Gravity: ({g.get('x',0)}, {g.get('y',0)})")
    lines.append(f"Timestep: {fmt(h.get('timestep', 0.01667), 5)}")
    if h.get("scenario_type"):
        lines.append(f"Type: {h['scenario_type']}")
    if h.get("difficulty") is not None:
        lines.append(f"Difficulty: {h['difficulty']}")
    stat = h.get("static_geometry") or []
    if stat:
        parts = []
        for sg in stat:
            if sg["type"] == "segment":
                parts.append(f"seg ({round(sg['p1']['x'])},{round(sg['p1']['y'])})-({round(sg['p2']['x'])},{round(sg['p2']['y'])})")
            elif sg["type"] == "circle":
                parts.append(f"peg ({round(sg['center']['x'])},{round(sg['center']['y'])}) r={round(sg['radius'])}")
        if parts:
            lines.append(f"Static: {'; '.join(parts)}")
    cons = h.get("constraints") or []
    if cons:
        parts = [f"{c['type']} {c['body_a']}->{c['body_b']}" for c in cons]
        lines.append(f"Constraints: {'; '.join(parts)}")
    lines.append("")
    return "\n".join(lines)

def fmt_frame(fr: dict) -> str:
    lines = [f"Frame {fr['frame']}: {fr.get('description','')}"]
    for o in fr["objects"]:
        p = o["position"]; v = o.get("velocity", {"x": 0, "y": 0})
        a = o.get("angle", 0); av = o.get("angular_velocity", 0)
        s = f"  obj_{o['id']}: pos=({fmt(p['x'],4)}, {fmt(p['y'],4)}), vel=({fmt(v['x'],4)}, {fmt(v['y'],4)})"
        if abs(a) > 0.001 or abs(av) > 0.001:
            s += f", a={fmt(a,4)}, av={fmt(av,4)}"
        lines.append(s)
    lines.append("")
    return "\n".join(lines)

# --- also try the *exact* training format from src/training/data_loader.py ---

def training_format_header(header: dict) -> str:
    """Mirrors src/training/data_loader.py:_format_header_text exactly."""
    lines = []
    lines.append(f"Scene: {header['description']}")
    gravity = header['gravity']
    lines.append(f"Gravity: ({gravity['x']}, {gravity['y']})")
    lines.append(f"Timestep: {header['timestep']:.5f}")
    if header.get('scenario_type'):
        lines.append(f"Type: {header['scenario_type']}")
    if header.get('difficulty'):
        lines.append(f"Difficulty: {header['difficulty']}")
    static_geom = header.get('static_geometry', [])
    if static_geom:
        gp = []
        for sg in static_geom:
            if sg['type'] == 'segment':
                p1, p2 = sg['p1'], sg['p2']
                gp.append(f"seg ({p1['x']:.0f},{p1['y']:.0f})-({p2['x']:.0f},{p2['y']:.0f})")
            elif sg['type'] == 'circle':
                c = sg['center']
                gp.append(f"peg ({c['x']:.0f},{c['y']:.0f}) r={sg['radius']:.0f}")
        if gp:
            lines.append(f"Static: {'; '.join(gp)}")
    cons = header.get('constraints', [])
    if cons:
        cp = [f"{c['type']} {c['body_a']}->{c['body_b']}" for c in cons]
        lines.append(f"Constraints: {'; '.join(cp)}")
    lines.append("")
    return "\n".join(lines)

def training_format_frame(frame: dict) -> str:
    lines = []
    lines.append(f"Frame {frame['frame']}: {frame['description']}")
    for obj in frame['objects']:
        pos = obj['position']; vel = obj['velocity']
        angle = obj.get('angle', 0); ang_vel = obj.get('angular_velocity', 0)
        s = f"  obj_{obj['id']}: pos=({pos['x']:.4f}, {pos['y']:.4f}), vel=({vel['x']:.4f}, {vel['y']:.4f})"
        if abs(angle) > 0.001 or abs(ang_vel) > 0.001:
            s += f", a={angle:.4f}, av={ang_vel:.4f}"
        lines.append(s)
    lines.append("")
    return "\n".join(lines)


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "billiards"
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    path = Path("/home/alexw/Projects/physics-llm-space/backend/examples") / f"{scenario}.jsonl"
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    header = json.loads(lines[0])
    initial = [json.loads(ln) for ln in lines[1:5]]

    # --- TS-style prompt (what the browser sends) ---
    prompt_ts = fmt_header(header) + "".join(fmt_frame(f) for f in initial) + "Predict next frame:"
    # --- Training-faithful prompt (what the model was trained on) ---
    prompt_train = training_format_header(header) + "".join(training_format_frame(f) for f in initial) + "Predict next frame:"

    print("\n=================== TS PROMPT (last 600 chars) ===================")
    print(prompt_ts[-600:])
    print("\n=================== TRAINING PROMPT (last 600 chars) =============")
    print(prompt_train[-600:])
    print("\n=================== DIFF ==========================================")
    if prompt_ts == prompt_train:
        print("PROMPTS ARE IDENTICAL")
    else:
        # Show first diverging line
        a = prompt_ts.splitlines(); b = prompt_train.splitlines()
        for i, (la, lb) in enumerate(zip(a, b)):
            if la != lb:
                print(f"  line {i}:")
                print(f"    TS    : {la!r}")
                print(f"    TRAIN : {lb!r}")
                break
        else:
            print(f"len mismatch: TS={len(a)} TRAIN={len(b)}")

    print("\n=================== LOADING MODEL =================================")
    from llama_cpp import Llama
    llm = Llama(
        model_path="/home/alexw/physics-llm-debug/lfm2-scenarios-Q4_K_M.gguf",
        n_ctx=8192,
        n_threads=8,
        verbose=False,
    )

    # Just the TS prompt (we already verified they're identical to training format)
    for label, prompt in (("TS", prompt_ts),):
        print(f"\n=================== GENERATING with {label} prompt ================")
        ctx = prompt
        for step in range(n_steps):
            out = llm(ctx, max_tokens=1024, temperature=0.1, top_p=0.95, stop=["Predict next frame:"])
            text = out["choices"][0]["text"]
            print(f"\n--- step {step+1} raw output ---")
            print(text)
            # Cut to first frame and append back
            m = list(re.finditer(r"Frame\s+\d+:", text))
            if not m:
                print("!!! NO 'Frame N:' in output — model didn't follow format")
                break
            start = m[0].start()
            end = m[1].start() if len(m) > 1 else len(text)
            first_frame = text[start:end].rstrip() + "\n"
            ctx = ctx.rsplit("Predict next frame:", 1)[0] + first_frame + "Predict next frame:"
            print(f"--- step {step+1} first frame extracted ---")
            print(first_frame)


if __name__ == "__main__":
    main()
