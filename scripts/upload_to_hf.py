#!/usr/bin/env python3
"""Upload scenario dataset to HuggingFace using upload_large_folder."""
import time
from huggingface_hub import HfApi

repo_id = "ICML-2026/physics-generaliztion"
api = HfApi()

folder = "/home/alexw/data_scenarios"

print(f"Uploading {folder} to {repo_id}...")
print(f"This includes train/ val/ manifest.json README.md")
start = time.time()

api.upload_large_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=folder,
    num_workers=8,
    ignore_patterns=["*.cache*", ".cache/**"],
    print_report_every=120,
)

elapsed = time.time() - start
print(f"\nDone in {elapsed/60:.1f} min")
