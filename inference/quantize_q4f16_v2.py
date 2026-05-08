"""Re-quantize patched q4 ONNX → q4f16, blocking Cast and other type-sensitive
nodes from fp16 conversion to avoid the gqa_attention_bias error."""
from __future__ import annotations
from pathlib import Path
import onnx
from onnxconverter_common import float16

ONNX_DIR = Path("/home/alexw/physics-llm-debug/lfm2-scenarios-onnx/onnx")
SRC = ONNX_DIR / "model_q4.onnx"
OUT = ONNX_DIR / "model_q4f16.onnx"
OUT_DATA = ONNX_DIR / "model_q4f16.onnx_data"

print(f"loading {SRC}")
m = onnx.load(str(SRC), load_external_data=True)

# Block Cast-related nodes from fp16 conversion. The base ONNX has
# gqa_attention_bias/Cast which expects fp32 input — converting to fp16
# breaks the type contract.
BLOCK_OPS = ['Cast', 'Range', 'Where', 'CumSum', 'NonZero']
m_q4f16 = float16.convert_float_to_float16(
    m,
    keep_io_types=True,
    op_block_list=BLOCK_OPS,
    disable_shape_infer=False,
)

print(f"saving {OUT}")
if OUT_DATA.exists():
    OUT_DATA.unlink()
onnx.save(m_q4f16, str(OUT), save_as_external_data=True,
          all_tensors_to_one_file=True, location="model_q4f16.onnx_data")
sz = OUT_DATA.stat().st_size // 1024 // 1024
print(f"OK · {sz} MB")
