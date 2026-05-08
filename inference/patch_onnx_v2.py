"""Smart patcher: maps fine-tuned safetensors weights to ONNX initializer names,
applying naming-convention translations and transposing MatMul weights."""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

BASE_ONNX = Path("/home/alexw/physics-llm-debug/lfm2-base-onnx/onnx/model.onnx")
MERGED = Path("/home/alexw/physics-llm-debug/lfm2-scenarios-merged")
OUT_DIR = Path("/home/alexw/physics-llm-debug/lfm2-scenarios-onnx")

# safetensors-name → ONNX-name translation rules (applied in order)
RULES = [
    # Top-level (ONNX moves final norm into last layer with a different name)
    (r"^model\.embedding_norm\.weight$", "model.layers.16.final_norm_layernorm.weight"),
    # Per-layer renames
    (r"^model\.layers\.(\d+)\.ffn_norm\.weight$", r"model.layers.\1.ffn_layernorm.weight"),
    (r"^model\.layers\.(\d+)\.operator_norm\.weight$", r"model.layers.\1.operator_layernorm.weight"),
    # MLP feed_forward.w1/2/3 -> mlp.gate/down/up_proj.MatMul (transposed)
    (r"^model\.layers\.(\d+)\.feed_forward\.w1\.weight$", r"model.layers.\1.mlp.gate_proj.MatMul.weight", "T"),
    (r"^model\.layers\.(\d+)\.feed_forward\.w2\.weight$", r"model.layers.\1.mlp.down_proj.MatMul.weight", "T"),
    (r"^model\.layers\.(\d+)\.feed_forward\.w3\.weight$", r"model.layers.\1.mlp.up_proj.MatMul.weight", "T"),
    # Conv layers
    (r"^model\.layers\.(\d+)\.conv\.in_proj\.weight$", r"model.layers.\1.conv.in_proj.MatMul.weight", "T"),
    (r"^model\.layers\.(\d+)\.conv\.out_proj\.weight$", r"model.layers.\1.conv.out_proj.MatMul.weight", "T"),
    # conv.conv.weight stays same (conv kernel, no transpose)
    # Self-attention
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", r"model.layers.\1.attn.q_proj.MatMul.weight", "T"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$", r"model.layers.\1.attn.k_proj.MatMul.weight", "T"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$", r"model.layers.\1.attn.v_proj.MatMul.weight", "T"),
    (r"^model\.layers\.(\d+)\.self_attn\.out_proj\.weight$", r"model.layers.\1.attn.o_proj.MatMul.weight", "T"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_layernorm\.weight$", r"model.layers.\1.attn.q_norm.layernorm.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_layernorm\.weight$", r"model.layers.\1.attn.k_norm.layernorm.weight"),
]


def translate(name: str) -> tuple[str | None, bool]:
    """Map safetensors name → onnx name. Returns (onnx_name, transpose)."""
    import re
    for rule in RULES:
        pat, repl = rule[0], rule[1]
        transpose = len(rule) > 2 and rule[2] == "T"
        if re.match(pat, name):
            return re.sub(pat, repl, name), transpose
    return None, False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_dir = OUT_DIR / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    print("loading base ONNX...")
    model = onnx.load(str(BASE_ONNX), load_external_data=True)
    inits = {init.name: init for init in model.graph.initializer}
    print(f"  {len(inits)} initializers")

    print("loading safetensors...")
    from safetensors import safe_open
    weights = {}
    for f in sorted(MERGED.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as st:
            for k in st.keys():
                weights[k] = st.get_tensor(k)
    print(f"  {len(weights)} tensors")

    name_to_idx = {init.name: i for i, init in enumerate(model.graph.initializer)}
    n_patch_direct = 0
    n_patch_translated = 0
    n_patch_transposed = 0
    n_skip_noonnx = 0
    n_skip_shape = 0
    skip_examples = []

    for st_name, t in weights.items():
        # Try direct first
        target = st_name if st_name in inits else None
        transpose = False
        if target is None:
            translated, transpose = translate(st_name)
            if translated and translated in inits:
                target = translated
        if target is None:
            n_skip_noonnx += 1
            if len(skip_examples) < 8:
                skip_examples.append(st_name)
            continue

        np_t = t.cpu().numpy().astype(np.float32)
        if transpose:
            np_t = np_t.T.copy()

        old_shape = tuple(inits[target].dims)
        if np_t.shape != old_shape:
            n_skip_shape += 1
            print(f"  shape mismatch {st_name} → {target}: {np_t.shape} vs {old_shape}")
            continue

        new_init = numpy_helper.from_array(np_t, name=target)
        idx = name_to_idx[target]
        model.graph.initializer[idx].CopyFrom(new_init)
        if st_name == target:
            n_patch_direct += 1
        else:
            if transpose:
                n_patch_transposed += 1
            else:
                n_patch_translated += 1

    print(f"\nPATCH SUMMARY:")
    print(f"  direct match patched: {n_patch_direct}")
    print(f"  renamed-only patched: {n_patch_translated}")
    print(f"  renamed+transposed:   {n_patch_transposed}")
    print(f"  no ONNX target found: {n_skip_noonnx}")
    print(f"  shape mismatches:     {n_skip_shape}")
    if skip_examples:
        print(f"  examples skipped: {skip_examples}")

    total_patched = n_patch_direct + n_patch_translated + n_patch_transposed
    print(f"\n  TOTAL patched: {total_patched} / {len(weights)} weights")

    print("\nsaving patched ONNX...")
    out_onnx = onnx_dir / "model.onnx"
    if out_onnx.exists():
        out_onnx.unlink()
        (onnx_dir / "model.onnx_data").unlink(missing_ok=True)
    onnx.save(model, str(out_onnx), save_as_external_data=True,
              all_tensors_to_one_file=True, location="model.onnx_data")
    print("OK", out_onnx, "size:", out_onnx.stat().st_size, "data:",
          (onnx_dir / "model.onnx_data").stat().st_size // 1024 // 1024, "MB")

    import shutil
    base_root = BASE_ONNX.parent.parent
    for f in ['config.json','tokenizer.json','tokenizer_config.json',
              'special_tokens_map.json','generation_config.json','chat_template.jinja']:
        src = base_root / f
        if src.exists():
            shutil.copy2(src, OUT_DIR / f)


if __name__ == "__main__":
    main()
