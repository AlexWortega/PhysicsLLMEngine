#!/usr/bin/env python3
"""
Validate generated physics simulation dataset and compute statistics.

Usage:
    python scripts/validate_dataset.py --data data

Output:
    - Scene counts per split
    - Unique seed verification (no duplicates)
    - Object count distribution (min, max, mean)
    - Shape type counts (circles vs rectangles)
    - Seed overlap check between splits
    - Storage size
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def get_directory_size(path):
    """Calculate total size of directory in bytes."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total += os.path.getsize(filepath)
            except (OSError, IOError):
                pass
    return total


def format_size(size_bytes):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def validate_split(split_dir):
    """Validate and compute statistics for a single split.

    Args:
        split_dir: Path to split directory (e.g., data/train)

    Returns:
        dict with statistics
    """
    split_dir = Path(split_dir)

    stats = {
        "total_scenes": 0,
        "total_objects": 0,
        "object_counts": [],
        "shape_types": Counter(),
        "seeds": set(),
        "invalid_files": [],
        "storage_bytes": 0,
    }

    # Iterate through all JSONL files
    for jsonl_file in split_dir.rglob("*.jsonl"):
        stats["total_scenes"] += 1
        stats["storage_bytes"] += jsonl_file.stat().st_size

        try:
            with open(jsonl_file) as f:
                header_line = f.readline()
                header = json.loads(header_line)

            # Extract metadata from header
            stats["seeds"].add(header["seed"])
            object_count = header.get("object_count", len(header.get("objects", [])))
            stats["total_objects"] += object_count
            stats["object_counts"].append(object_count)

            # Count shape types from objects list
            for obj in header.get("objects", []):
                obj_type = obj.get("type", "unknown")
                stats["shape_types"][obj_type] += 1

        except (json.JSONDecodeError, KeyError, IOError) as e:
            stats["invalid_files"].append({
                "file": str(jsonl_file),
                "error": str(e)
            })

    # Compute distribution statistics
    counts = stats["object_counts"]
    if counts:
        stats["object_count_stats"] = {
            "min": min(counts),
            "max": max(counts),
            "mean": round(sum(counts) / len(counts), 2),
        }
    else:
        stats["object_count_stats"] = {"min": 0, "max": 0, "mean": 0}

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Validate generated dataset and compute statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        help="Data directory path",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data directory not found: {data_path}")
        return

    all_stats = {}
    all_seeds = {}
    total_storage = 0

    # Validate each split
    for split in ["train", "val", "test"]:
        split_dir = data_path / split
        if not split_dir.exists():
            print(f"\n{split.upper()}: Not found")
            continue

        print(f"\nValidating {split}...")
        stats = validate_split(split_dir)
        all_stats[split] = stats
        all_seeds[split] = stats["seeds"]
        total_storage += stats["storage_bytes"]

        print(f"\n{split.upper()}:")
        print(f"  Scenes: {stats['total_scenes']:,}")
        print(f"  Unique seeds: {len(stats['seeds']):,}")
        print(f"  Objects: {stats['total_objects']:,} total")
        ocs = stats["object_count_stats"]
        print(f"  Object count range: {ocs['min']}-{ocs['max']} (mean: {ocs['mean']})")
        print(f"  Shape types: {dict(stats['shape_types'])}")
        print(f"  Storage: {format_size(stats['storage_bytes'])}")

        if stats["invalid_files"]:
            print(f"  WARNING: {len(stats['invalid_files'])} invalid files")
            for inv in stats["invalid_files"][:3]:
                print(f"    - {inv['file']}: {inv['error']}")
            if len(stats["invalid_files"]) > 3:
                print(f"    ... and {len(stats['invalid_files']) - 3} more")

    # Overall statistics
    if all_stats:
        print(f"\n{'=' * 60}")
        print("OVERALL:")
        print("=" * 60)

        total_scenes = sum(s["total_scenes"] for s in all_stats.values())
        total_objects = sum(s["total_objects"] for s in all_stats.values())
        total_unique_seeds = sum(len(s["seeds"]) for s in all_stats.values())

        print(f"  Total scenes: {total_scenes:,}")
        print(f"  Total objects: {total_objects:,}")
        print(f"  Total storage: {format_size(total_storage)}")

        # Check for seed overlap between splits
        print(f"\n  Seed overlap check:")
        overlap_found = False

        # Check all pairs of splits
        split_names = list(all_seeds.keys())
        for i, split1 in enumerate(split_names):
            for split2 in split_names[i + 1:]:
                overlap = all_seeds[split1] & all_seeds[split2]
                if overlap:
                    overlap_found = True
                    print(f"    FAIL: {split1} and {split2} share {len(overlap)} seeds!")
                    print(f"           Examples: {sorted(list(overlap))[:5]}")
                else:
                    print(f"    PASS: {split1} and {split2} have no overlap")

        if not overlap_found:
            print(f"\n  Seed overlap check: PASS (no overlap between splits)")
        else:
            print(f"\n  Seed overlap check: FAIL (overlap detected!)")

        # Shape type distribution overall
        overall_shapes = Counter()
        for s in all_stats.values():
            overall_shapes.update(s["shape_types"])
        print(f"\n  Overall shape distribution: {dict(overall_shapes)}")

        # Object count distribution
        all_counts = []
        for s in all_stats.values():
            all_counts.extend(s["object_counts"])
        if all_counts:
            print(f"  Overall object count: {min(all_counts)}-{max(all_counts)} "
                  f"(mean: {sum(all_counts) / len(all_counts):.2f})")


if __name__ == "__main__":
    main()
