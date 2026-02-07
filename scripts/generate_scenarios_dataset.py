#!/usr/bin/env python3
"""
Generate 1M diverse physics scenario samples.

Split strategy:
  - SEEN scenarios (24 types): appear in train + val
  - UNSEEN scenarios (6 types): appear ONLY in val (held-out for OOD evaluation)

Unseen (simple + complex mix):
  Simple:  pong, bowling, ramp_roll
  Complex: angry_birds, hourglass, newtons_cradle

Distribution:
  Train: 900,000 samples / 24 seen types = 37,500 per type
  Val:   100,000 samples / 30 all types  =  3,333 per type

Seed ranges (no overlap with original dataset 0-99999 or OOD 200000+):
  Train: 1,000,000+
  Val:   5,000,000+

Usage:
  python scripts/generate_scenarios_dataset.py --workers -1
  python scripts/generate_scenarios_dataset.py --split train --workers 16
  python scripts/generate_scenarios_dataset.py --split val --workers 16
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

sys.path.insert(0, "/home/alexw")

from tqdm import tqdm

from src.physics.scenario_generator import generate_scenario, SCENARIO_TYPES
from src.physics.scenario_registry import list_scenarios
from src.data.exporter import export_simulation

# ── Unseen scenarios: only in val ──────────────────────────────────
UNSEEN_SIMPLE = ["pong", "bowling", "ramp_roll"]
UNSEEN_COMPLEX = ["angry_birds", "hourglass", "newtons_cradle"]
UNSEEN = set(UNSEEN_SIMPLE + UNSEEN_COMPLEX)

SEEN = sorted([s for s in SCENARIO_TYPES if s not in UNSEEN])

assert len(SEEN) == 24, f"Expected 24 seen, got {len(SEEN)}"
assert len(UNSEEN) == 6, f"Expected 6 unseen, got {len(UNSEEN)}"

# ── Split config ───────────────────────────────────────────────────
TRAIN_PER_TYPE = 37_500         # 24 types × 37,500 = 900,000
VAL_PER_TYPE = 3_334            # 30 types × 3,334  = 100,020 (~100K)

SEED_BASE_TRAIN = 1_000_000
SEED_BASE_VAL = 5_000_000

NUM_FRAMES = 200


def build_tasks(split):
    """Build (seed, scenario_type, output_path) tuples for a split."""
    tasks = []
    if split == "train":
        scenario_list = SEEN
        per_type = TRAIN_PER_TYPE
        seed_base = SEED_BASE_TRAIN
    else:
        scenario_list = sorted(SCENARIO_TYPES)  # all 30
        per_type = VAL_PER_TYPE
        seed_base = SEED_BASE_VAL

    for type_idx, stype in enumerate(scenario_list):
        for i in range(per_type):
            seed = seed_base + type_idx * per_type + i
            # Bucket: scenario_type/seed // 1000
            bucket = f"{seed // 1000:07d}"
            path = f"data_scenarios/{split}/{stype}/{bucket}/scene_{seed:08d}.jsonl"
            tasks.append((seed, stype, path))
    return tasks


def worker(args):
    """Generate one scenario scene."""
    seed, stype, output_path = args

    out = Path(output_path)
    if out.exists():
        return {"seed": seed, "type": stype, "skipped": True, "success": True}

    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        sim, meta = generate_scenario(seed=seed, scenario_type=stype)
        export_simulation(sim, NUM_FRAMES, str(out), seed=seed, metadata=meta)
        return {
            "seed": seed, "type": stype,
            "objects": len(sim.bodies),
            "success": True, "skipped": False,
        }
    except Exception as e:
        return {
            "seed": seed, "type": stype,
            "success": False, "skipped": False,
            "error": str(e),
        }


def generate_split(split, num_workers):
    """Generate all scenes for a split."""
    tasks = build_tasks(split)
    total = len(tasks)

    n_types = len(SEEN) if split == "train" else len(SCENARIO_TYPES)
    per = TRAIN_PER_TYPE if split == "train" else VAL_PER_TYPE
    print(f"\n{'='*60}")
    print(f"Split: {split}")
    print(f"Scenario types: {n_types} ({'seen only' if split == 'train' else 'all 30 (seen + unseen)'})")
    print(f"Per type: {per:,}")
    print(f"Total scenes: {total:,}")
    print(f"Workers: {num_workers}")
    print(f"{'='*60}")

    generated = 0
    skipped = 0
    failed = 0
    total_objects = 0
    errors = []

    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(worker, tasks, chunksize=64),
                           total=total, desc=split, unit="scene",
                           miniters=500, smoothing=0.1):
            if not result["success"]:
                failed += 1
                if len(errors) < 20:
                    errors.append(result)
            elif result.get("skipped"):
                skipped += 1
            else:
                generated += 1
                total_objects += result.get("objects", 0)

    print(f"\n{split} summary:")
    print(f"  Generated: {generated:,}")
    print(f"  Skipped:   {skipped:,}")
    print(f"  Failed:    {failed:,}")
    if generated > 0:
        print(f"  Avg objects/scene: {total_objects / generated:.1f}")
    if errors:
        print(f"  First errors:")
        for e in errors[:5]:
            print(f"    seed={e['seed']} type={e['type']}: {e.get('error','?')}")

    return generated, skipped, failed


def write_manifest(output_dir="data_scenarios"):
    """Write manifest with split info."""
    manifest = {
        "seen_scenarios": SEEN,
        "unseen_scenarios": sorted(UNSEEN),
        "unseen_simple": UNSEEN_SIMPLE,
        "unseen_complex": UNSEEN_COMPLEX,
        "train_per_type": TRAIN_PER_TYPE,
        "val_per_type": VAL_PER_TYPE,
        "train_total": TRAIN_PER_TYPE * len(SEEN),
        "val_total": VAL_PER_TYPE * len(SCENARIO_TYPES),
        "num_frames": NUM_FRAMES,
        "seed_base_train": SEED_BASE_TRAIN,
        "seed_base_val": SEED_BASE_VAL,
    }
    path = Path(output_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 1M diverse physics scenario samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--workers", type=int, default=-1,
                        help="Parallel workers (-1 = all cores)")
    args = parser.parse_args()

    num_workers = args.workers if args.workers > 0 else cpu_count()

    splits = ["train", "val"] if args.split == "all" else [args.split]

    start = time.time()
    totals = {"generated": 0, "skipped": 0, "failed": 0}

    for split in splits:
        g, s, f = generate_split(split, num_workers)
        totals["generated"] += g
        totals["skipped"] += s
        totals["failed"] += f

    write_manifest()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Total generated: {totals['generated']:,}")
    print(f"Total skipped:   {totals['skipped']:,}")
    print(f"Total failed:    {totals['failed']:,}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    if totals["generated"] > 0:
        rate = totals["generated"] / elapsed
        print(f"Rate: {rate:.0f} scenes/sec")


if __name__ == "__main__":
    main()
