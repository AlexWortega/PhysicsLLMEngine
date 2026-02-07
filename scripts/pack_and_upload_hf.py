#!/usr/bin/env python3
"""Pack scenario dataset into tar.gz files per scenario type and upload to HF."""
import os
import subprocess
import time
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "ICML-2026/physics-generaliztion"
DATA_DIR = Path("/home/alexw/data_scenarios")
PACK_DIR = Path("/home/alexw/data_scenarios_packed")

api = HfApi()


def pack_split(split):
    """Pack each scenario type into a tar.gz."""
    split_dir = DATA_DIR / split
    out_dir = PACK_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    print(f"\n{split}: {len(scenario_dirs)} scenario types to pack")

    for sd in scenario_dirs:
        name = sd.name
        tar_path = out_dir / f"{name}.tar.gz"
        if tar_path.exists():
            print(f"  {name}: already packed, skip")
            continue

        n_files = sum(1 for _ in sd.rglob("*.jsonl"))
        print(f"  {name}: packing {n_files} files...", end=" ", flush=True)
        start = time.time()
        subprocess.run(
            ["tar", "czf", str(tar_path), "-C", str(split_dir), name],
            check=True, capture_output=True,
        )
        elapsed = time.time() - start
        size_mb = tar_path.stat().st_size / 1024 / 1024
        print(f"{size_mb:.0f}MB in {elapsed:.0f}s")


def upload():
    """Upload packed files + metadata to HF."""
    print("\n" + "=" * 60)
    print("Uploading to HuggingFace...")
    print("=" * 60)

    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(PACK_DIR),
        num_workers=4,
        ignore_patterns=[".cache/**"],
    )

    # Also upload README and manifest from original dir
    for fname in ["README.md", "manifest.json"]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=REPO_ID,
                repo_type="dataset",
            )
            print(f"Uploaded {fname}")


def main():
    start = time.time()

    for split in ["train", "val"]:
        pack_split(split)

    # Copy manifest
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(DATA_DIR / "manifest.json", PACK_DIR / "manifest.json")

    total_size = sum(f.stat().st_size for f in PACK_DIR.rglob("*.tar.gz")) / 1024**3
    n_files = sum(1 for _ in PACK_DIR.rglob("*.tar.gz"))
    print(f"\nPacked: {n_files} tar.gz files, {total_size:.1f}GB total")

    upload()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
