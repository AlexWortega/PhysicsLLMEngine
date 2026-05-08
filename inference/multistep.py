"""Multi-step rollout — track per-step movement, drift, parse-failures.

Usage: python multistep.py <scenario> <n_steps>
Prints per-step: max-pos-delta, mean-pos-delta, n-objects-parsed, top mover.
"""
from __future__ import annotations
import json, sys, re, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from repro import fmt_header, fmt_frame  # reuse identical formatting

scenario = sys.argv[1] if len(sys.argv) > 1 else "billiards"
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20

ex = Path("/home/alexw/Projects/physics-llm-space/backend/examples") / f"{scenario}.jsonl"
lines = [ln for ln in ex.read_text().splitlines() if ln.strip()]
header = json.loads(lines[0])
initial = [json.loads(ln) for ln in lines[1:5]]
n_obj = header["object_count"]

ctx = fmt_header(header) + "".join(fmt_frame(f) for f in initial)

from llama_cpp import Llama
print(f"loading model... scenario={scenario} n_obj={n_obj} n_steps={n_steps}")
llm = Llama(
    model_path="/home/alexw/physics-llm-debug/lfm2-scenarios-Q4_K_M.gguf",
    n_ctx=8192, n_threads=8, verbose=False,
)

OBJ_RE = re.compile(
    r"obj_(\d+):\s*pos=\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\),\s*vel=\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)"
)

# state from last GROUND-TRUTH frame (frame 4)
prev = {o["id"]: (o["position"]["x"], o["position"]["y"]) for o in initial[-1]["objects"]}
last_idx = initial[-1]["frame"]
parse_fails = 0
zero_motion_steps = 0
diverged = False
t0 = time.time()

for step in range(n_steps):
    prompt = ctx + "Predict next frame:"
    out = llm(prompt, max_tokens=1024, temperature=0.1, top_p=0.95, stop=["Predict next frame:"])
    text = out["choices"][0]["text"]

    # Cap context to keep + 4 frames so it never blows up
    ms = list(re.finditer(r"Frame\s+\d+:", text))
    if ms:
        first = text[ms[0].start(): ms[1].start() if len(ms) > 1 else len(text)]
        had_header = True
    else:
        first = text  # model dropped the header — try to parse anyway
        had_header = False

    # Parse positions (works even without "Frame N:" header)
    new = {}
    for m in OBJ_RE.finditer(first):
        i = int(m.group(1))
        if i < n_obj:
            new[i] = (float(m.group(2)), float(m.group(3)))
    if not had_header:
        print(f"step {step+1:2d} | DRIFT: no 'Frame N:' header (parsed {len(new)}/{n_obj} via obj_ regex)")

    parsed = len(new)
    if parsed == 0:
        parse_fails += 1
    deltas = []
    for i, (x, y) in new.items():
        if i in prev:
            dx = x - prev[i][0]; dy = y - prev[i][1]
            deltas.append((i, (dx**2 + dy**2) ** 0.5, dx, dy))

    if not deltas:
        max_d = mean_d = 0.0
        top = "-"
    else:
        max_d = max(d[1] for d in deltas)
        mean_d = sum(d[1] for d in deltas) / len(deltas)
        top = max(deltas, key=lambda d: d[1])
        top = f"obj_{top[0]} dist={top[1]:.3f} dx={top[2]:.2f} dy={top[3]:.2f}"

    if max_d < 0.01 and parsed > 0:
        zero_motion_steps += 1

    print(f"step {step+1:2d} | parsed {parsed:2d}/{n_obj} | max-dist {max_d:7.3f} | mean {mean_d:6.3f} | top: {top}")

    # check for divergence (positions exploding or NaN)
    for x, y in new.values():
        if abs(x) > 5000 or abs(y) > 5000:
            diverged = True

    # update prev — fall back to previous values for missing obj_ids
    full = dict(prev)
    full.update(new)
    prev = full
    last_idx += 1

    # roll the model's actual emitted frame back into context. If model
    # dropped the header, synthesize one so the next prompt stays in-dist.
    if had_header:
        append = first.rstrip() + "\n"
    else:
        objs = []
        for i in range(n_obj):
            x, y = (new.get(i) or prev.get(i) or (0, 0))
            objs.append(f"  obj_{i}: pos=({x:.4f}, {y:.4f}), vel=(0.0000, 0.0000)")
        append = f"Frame {last_idx}: All objects are in motion.\n" + "\n".join(objs) + "\n"

    ctx_chunks = ctx.split("Frame ", 1)
    head = ctx_chunks[0]
    body = "Frame " + ctx_chunks[1] if len(ctx_chunks) > 1 else ""
    body_frames = body.split("\nFrame ") if body else []
    if body_frames:
        body_frames[1:] = ["Frame " + f for f in body_frames[1:]]
    body_frames.append(append)
    if len(body_frames) > 4:
        body_frames = body_frames[-4:]
    ctx = head + "".join(body_frames)

dt = time.time() - t0
print(f"\nsummary: {n_steps} steps in {dt:.1f}s ({n_steps/dt:.2f} steps/s)")
print(f"parse-failures: {parse_fails}")
print(f"zero-motion steps: {zero_motion_steps}")
print(f"diverged (|pos|>5000): {diverged}")
