"""Measure prefill+gen times for 1-frame vs 4-frame prompts on q4 ONNX.
Mimics what the browser will see post-quick-wins."""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from repro import fmt_header, fmt_frame

ONNX = "/home/alexw/physics-llm-debug/lfm2-scenarios-onnx/onnx/model_q4.onnx"
MERGED = "/home/alexw/physics-llm-debug/lfm2-scenarios-merged"
SCN = "/home/alexw/Projects/physics-llm-space/backend/examples/billiards.jsonl"

lines = [ln for ln in Path(SCN).read_text().splitlines() if ln.strip()]
header = json.loads(lines[0])
frames = [json.loads(ln) for ln in lines[1:5]]

# Two prompts to compare
prompt_4f = fmt_header(header) + "".join(fmt_frame(f) for f in frames) + "Predict next frame:"
prompt_1f = fmt_header(header) + fmt_frame(frames[0]) + "Predict next frame:"

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MERGED)

import onnxruntime as ort
print(f"loading q4 ONNX...")
sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
input_names = [i.name for i in sess.get_inputs()]
out_names = [o.name for o in sess.get_outputs()]


def setup_inputs(seq_len: int):
    inp = {
        'input_ids': np.zeros((1, seq_len), dtype=np.int64),
        'attention_mask': np.ones((1, seq_len), dtype=np.int64),
        'num_logits_to_keep': np.array(1, dtype=np.int64),
    }
    for n in input_names:
        if 'past_conv' in n: inp[n] = np.zeros((1,1024,3), dtype=np.float32)
        elif 'past_key_values' in n: inp[n] = np.zeros((1,8,0,64), dtype=np.float32)
    return inp


for label, prompt in (("4-frame", prompt_4f), ("1-frame", prompt_1f)):
    ids = tok(prompt, return_tensors="np")["input_ids"]
    print(f"\n=== {label}: prompt = {len(prompt)} chars, {ids.shape[1]} tokens ===")
    inp = setup_inputs(ids.shape[1])
    inp['input_ids'] = ids.astype(np.int64)
    inp['attention_mask'] = np.ones_like(ids, dtype=np.int64)

    t0 = time.time()
    out = sess.run(None, inp)
    pf = time.time() - t0
    pf_tps = ids.shape[1] / pf
    print(f"prefill: {pf:.2f}s = {pf_tps:.1f} tok/s")

    # 30 gen tokens
    t1 = time.time()
    n_gen = 30
    seq_len = ids.shape[1]
    next_id = int(np.argmax(out[0][0, -1, :]))
    for step in range(n_gen):
        nxt_inp = {
            'input_ids': np.array([[next_id]], dtype=np.int64),
            'attention_mask': np.ones((1, seq_len + step + 1), dtype=np.int64),
            'num_logits_to_keep': np.array(1, dtype=np.int64),
        }
        for n in input_names:
            if 'past_key_values' in n or n.startswith('past_'):
                pres = n.replace('past_key_values', 'present').replace('past_', 'present_')
                if pres in out_names:
                    nxt_inp[n] = out[out_names.index(pres)]
                else:
                    nxt_inp[n] = inp[n]
        out = sess.run(None, nxt_inp)
        next_id = int(np.argmax(out[0][0, -1, :]))
    gen = time.time() - t1
    gen_tps = n_gen / gen
    print(f"gen 30 tokens: {gen:.2f}s = {gen_tps:.1f} tok/s")
    total = pf + gen
    print(f"total {ids.shape[1]+n_gen} tok in {total:.2f}s = {(ids.shape[1]+n_gen)/total:.1f} avg tok/s")
