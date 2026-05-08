"""Merge LoRA adapter into LFM2-350M base, then export to ONNX (fp16 + q4).
Upload to HF as AlexWortega/lfm2-scenarios-ONNX for transformers.js / WebGPU.
"""
from __future__ import annotations
import os, subprocess, sys, shutil
from pathlib import Path

WORK = Path("/home/alexw/physics-llm-debug")
MERGED = WORK / "lfm2-scenarios-merged"
ONNX_OUT = WORK / "lfm2-scenarios-onnx"
BASE = "LiquidAI/LFM2-350M"
ADAPTER = "AlexWortega/lfm2-scenarios"  # subfolder=final
ADAPTER_SUBFOLDER = "final"


def step(msg): print(f"\n=== {msg} ===", flush=True)


def merge_lora():
    step("merging LoRA adapter into base")
    if MERGED.exists():
        print(f"already exists at {MERGED}; skipping merge")
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    print(f"loading base: {BASE}")
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="float32")
    print(f"loading adapter: {ADAPTER}/{ADAPTER_SUBFOLDER}")
    model = PeftModel.from_pretrained(base, ADAPTER, subfolder=ADAPTER_SUBFOLDER)
    print("merge_and_unload()...")
    merged = model.merge_and_unload()
    print(f"saving merged to {MERGED}")
    merged.save_pretrained(str(MERGED), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(ADAPTER, subfolder=ADAPTER_SUBFOLDER)
    tok.save_pretrained(str(MERGED))
    print("DONE merge")


def export_onnx():
    step("exporting to ONNX via optimum-cli")
    if ONNX_OUT.exists() and (ONNX_OUT / "model.onnx").exists():
        print("already exported; skipping")
        return
    cmd = [
        "/home/alexw/physics_venv/bin/optimum-cli", "export", "onnx",
        "--model", str(MERGED),
        "--task", "text-generation-with-past",
        "--opset", "18",
        str(ONNX_OUT),
    ]
    print(" ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        print(f"optimum export FAILED with code {res.returncode}")
        sys.exit(1)
    print("DONE onnx export")


def quantize_onnx():
    """Produce q4f16 / q8 quantized variants suitable for transformers.js."""
    step("quantizing ONNX to q4f16 / q8")
    src = ONNX_OUT / "model.onnx"
    if not src.exists():
        print("no source model.onnx — skip"); return
    # transformers.js looks for these specific filenames in onnx/
    targets = [
        ("model_q4f16.onnx", "q4f16"),
        ("model_q4.onnx", "q4"),
        ("model_quantized.onnx", "uint8"),
    ]
    for fname, mode in targets:
        out = ONNX_OUT / fname
        if out.exists():
            print(f"  {fname} exists, skip")
            continue
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            print(f"  building {fname} ({mode})")
            if mode in ("q4", "q4f16"):
                # ORT 1.18+ has matmul_4bits_quantizer
                from onnxruntime.quantization.matmul_4bits_quantizer import (
                    MatMul4BitsQuantizer,
                )
                quant = MatMul4BitsQuantizer(
                    str(src),
                    block_size=32,
                    is_symmetric=True,
                    accuracy_level=4 if mode == "q4f16" else None,
                )
                quant.process()
                quant.model.save_model_to_file(str(out), use_external_data_format=False)
            else:  # uint8 dynamic
                quantize_dynamic(str(src), str(out), weight_type=QuantType.QUInt8)
            print(f"  {fname} OK ({out.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"  {fname} FAILED: {e}")
    print("DONE quantize")


def main():
    merge_lora()
    export_onnx()
    quantize_onnx()
    step("layout in onnx/ for HF (transformers.js convention)")
    onnx_dir = ONNX_OUT / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    for f in ONNX_OUT.glob("model*.onnx*"):
        if f.parent == ONNX_OUT:
            shutil.move(str(f), onnx_dir / f.name)
    print("ready to upload from", ONNX_OUT)


if __name__ == "__main__":
    main()
