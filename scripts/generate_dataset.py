#!/usr/bin/env python3
"""
Generate 100K+ physics simulation scenes in parallel.

Usage:
    python scripts/generate_dataset.py --split all --workers -1
    python scripts/generate_dataset.py --split train --workers 8

Seed ranges (non-overlapping):
    train: 0-79999     (80K scenes)
    val:   80000-89999 (10K scenes)
    test:  90000-99999 (10K scenes)

Output structure:
    data/
    ├── train/
    │   ├── 000000/  (seeds 0-999)
    │   │   ├── scene_000000.jsonl
    │   │   └── ...
    │   ├── 000001/  (seeds 1000-1999)
    │   └── ...
    ├── val/
    │   └── 000080/  (seeds 80000-80999)
    │   └── ...
    └── test/
        └── 000090/  (seeds 90000-90999)
        └── ...
"""

import argparse
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

from src.physics.scene_generator import generate_scene
from src.data.exporter import export_simulation


# Seed ranges for each split (non-overlapping, deterministic)
SPLITS = {
    "train": range(0, 80000),      # 80K scenes
    "val": range(80000, 90000),    # 10K scenes
    "test": range(90000, 100000),  # 10K scenes
}


def generate_single_scene(args):
    """Worker function for parallel scene generation.

    Args:
        args: tuple of (seed, output_dir, num_frames)

    Returns:
        dict with generation metadata:
            - seed: the seed used
            - object_count: number of objects (if success)
            - file_path: output file path (if success)
            - success: True if generated successfully
            - skipped: True if file already existed
            - error: error message (if failed)
    """
    seed, output_dir, num_frames = args

    # Bucket into subdirectories: seed // 1000 (e.g., seed 42 -> 000000/)
    bucket = f"{seed // 1000:06d}"
    output_path = Path(output_dir) / bucket / f"scene_{seed:06d}.jsonl"

    # Skip if already exists (resumable generation)
    if output_path.exists():
        return {"seed": seed, "skipped": True, "success": True}

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Generate scene using Phase 1 infrastructure
        sim = generate_scene(seed=seed)

        # Export to JSONL using Phase 1 exporter
        export_simulation(sim, num_frames, str(output_path), seed=seed)

        return {
            "seed": seed,
            "object_count": len(sim.bodies),
            "file_path": str(output_path),
            "success": True,
            "skipped": False,
        }
    except Exception as e:
        return {
            "seed": seed,
            "success": False,
            "skipped": False,
            "error": str(e),
        }


def generate_split(split_name, output_base, num_workers, num_frames):
    """Generate all scenes for a single split.

    Args:
        split_name: 'train', 'val', or 'test'
        output_base: base output directory (e.g., 'data')
        num_workers: number of parallel workers
        num_frames: frames per scene

    Returns:
        tuple of (results list, failed list)
    """
    seeds = list(SPLITS[split_name])
    output_dir = Path(output_base) / split_name

    # Prepare arguments for each worker
    args_list = [(seed, output_dir, num_frames) for seed in seeds]

    results = []
    failed = []
    skipped = 0
    total_objects = 0

    with Pool(processes=num_workers) as pool:
        # imap_unordered returns results as they complete (faster than map)
        for result in tqdm(
            pool.imap_unordered(generate_single_scene, args_list),
            total=len(seeds),
            desc=f"Generating {split_name}",
        ):
            results.append(result)
            if not result["success"]:
                failed.append(result)
            if result.get("skipped"):
                skipped += 1
            if result.get("object_count"):
                total_objects += result["object_count"]

    # Report summary
    generated = len(results) - skipped - len(failed)
    print(f"\n{split_name}: {len(seeds)} total seeds")
    print(f"  Generated: {generated}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {len(failed)}")
    if total_objects > 0:
        print(f"  Total objects: {total_objects}")

    # Report failures
    if failed:
        print(f"\nFailed seeds in {split_name}:")
        for f in failed[:10]:  # Show first 10
            print(f"  seed {f['seed']}: {f.get('error', 'unknown error')}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    return results, failed


def main():
    parser = argparse.ArgumentParser(
        description="Generate 100K+ physics simulation scenes in parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Base output directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Number of workers (-1 for all cores)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=200,
        help="Frames per scene",
    )
    args = parser.parse_args()

    # Determine worker count
    num_workers = args.workers if args.workers > 0 else cpu_count()
    print(f"Using {num_workers} workers")

    # Determine which splits to generate
    if args.split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]

    start_time = time.time()

    all_failed = []
    for split in splits:
        print(f"\n{'=' * 60}")
        print(f"Generating {split} split ({len(SPLITS[split])} scenes)")
        print("=" * 60)

        _, failed = generate_split(split, args.output, num_workers, args.frames)
        all_failed.extend(failed)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
    print(f"Total failures: {len(all_failed)}")

    if all_failed:
        print("\nFailed seeds (all splits):")
        for f in all_failed:
            print(f"  seed {f['seed']}: {f.get('error', 'unknown error')}")


if __name__ == "__main__":
    main()
