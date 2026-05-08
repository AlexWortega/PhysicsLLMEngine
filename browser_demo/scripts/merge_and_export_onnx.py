#!/usr/bin/env python3
"""Merge LoRA into LFM2-350M base, export to ONNX, quantize, upload."""
import shutil
import subprocess
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "LiquidAI/LFM2-350M"
ADAPTER = "/home/alexw/checkpoints/lfm2-scenarios/final"
WORK = Path("/tmp/lfm2_onnx_work")
MERGED = WORK / "merged"
ONNX_OUT = WORK / "onnx"
TARGET_REPO = "AlexWortega/lfm2-scenarios-ONNX"

WORK.mkdir(exist_ok=True)


def merge():
    if (MERGED / "config.json").exists():
        print("merged already exists, skip", flush=True)
        return
    print(f"loading base {BASE} ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    print(f"loading adapter {ADAPTER} ...", flush=True)
    model = PeftModel.from_pretrained(base, ADAPTER)
    print("merging...", flush=True)
    merged = model.merge_and_unload()
    MERGED.mkdir(parents=True, exist_ok=True)
    print(f"saving to {MERGED} ...", flush=True)
    merged.save_pretrained(str(MERGED), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    tok.save_pretrained(str(MERGED))
    print("merge done", flush=True)


def export_onnx():
    if (ONNX_OUT / "onnx" / "model.onnx").exists():
        print("onnx already exists, skip", flush=True)
        return
    print("exporting to ONNX via optimum-cli...", flush=True)
    cmd = [
        "/home/alexw/physics_venv/bin/optimum-cli", "export", "onnx",
        "--model", str(MERGED),
        "--task", "text-generation-with-past",
        "--opset", "14",
        "--no-post-process",
        str(ONNX_OUT),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print("ONNX export done", flush=True)


def quantize():
    """Quantize to q4 for browser. Use onnxruntime's quant tools."""
    src = ONNX_OUT / "model.onnx"
    if not src.exists():
        # Some exports nest under onnx/
        cand = ONNX_OUT / "onnx" / "model.onnx"
        if cand.exists():
            src = cand
    print(f"quantizing {src} ...", flush=True)
    from onnxruntime.quantization import matmul_4bits_quantizer
    from onnx import load_model

    out_q4 = src.parent / "model_q4.onnx"
    if out_q4.exists():
        print("q4 exists, skip", flush=True)
        return
    model = load_model(str(src))
    quant_config = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=32, is_symmetric=False
    )
    quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(model, algo_config=quant_config)
    quant.process()
    quant.model.save_model_to_file(str(out_q4), use_external_data_format=True)
    print(f"q4 saved to {out_q4}", flush=True)


def upload():
    print(f"\ncreating {TARGET_REPO} ...", flush=True)
    create_repo(TARGET_REPO, repo_type="model", private=False, exist_ok=True)
    api = HfApi()
    print("uploading...", flush=True)
    api.upload_folder(
        repo_id=TARGET_REPO,
        repo_type="model",
        folder_path=str(ONNX_OUT),
        commit_message="Upload merged ONNX (q4)",
    )
    print(f"https://huggingface.co/{TARGET_REPO}", flush=True)


def main():
    t0 = time.time()
    merge()
    export_onnx()
    quantize()
    upload()
    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
