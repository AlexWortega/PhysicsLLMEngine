# Inference & Debug Scripts

Local inference and evaluation scripts for the fine-tuned LFM2-350M physics predictor.

## Scripts

| File | Purpose |
|------|---------|
| `repro.py` | Single-step local repro — mirrors browser TS prompt format |
| `multistep.py` | Multi-step rollout with per-step drift/parse diagnostics |
| `bench.py` | Benchmark prefill+gen speed: 1-frame vs 4-frame prompts (ONNX) |
| `make_gifs.py` | Run all 30 demo scenarios × N frames, render animated GIFs |
| `patch_onnx_v2.py` | Patch fine-tuned LoRA weights into base LFM2 ONNX graph |
| `convert_to_onnx.py` | Export merged model to ONNX |
| `quantize_q4f16_v2.py` | Q4F16 re-quantization attempt (blocked by Cast op constraint) |

## Quick start

```bash
pip install llama-cpp-python onnxruntime matplotlib pillow
# Download GGUF from HuggingFace: AlexWortega/lfm2-scenarios-ONNX
python repro.py billiards 3
python multistep.py orbit 20
python make_gifs.py --frames 200 --out ./gifs
```
