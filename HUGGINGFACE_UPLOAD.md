# Uploading to Hugging Face

## Prerequisites

```bash
pip install huggingface_hub
```

Get your Hugging Face token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with `write` permissions
3. Set it as environment variable:

```bash
export HF_TOKEN="hf_..."
```

## Upload Model (LoRA Adapter)

```bash
python scripts/upload_to_huggingface.py \
  --type model \
  --checkpoint checkpoints/lfm2-scenarios/stage2/checkpoint-1000 \
  --repo alexwortega/LFM_Physics350M \
  --token $HF_TOKEN
```

This uploads:
- `adapter_config.json`
- `adapter_model.safetensors`
- `config.json`
- Tokenizer files
- `README.md` (MODEL_CARD.md)

## Upload Dataset

⚠️ **Warning:** The full dataset is ~582 GB. Uploading will take hours.

```bash
python scripts/upload_to_huggingface.py \
  --type dataset \
  --data-dir data_scenarios \
  --repo alexwortega/physics-scenarios \
  --token $HF_TOKEN
```

### Alternative: Upload Dataset Using `datasets` Library

For better compression and streaming support:

```python
from datasets import Dataset, DatasetDict
import json
from pathlib import Path

def load_scenes(split_dir):
    """Load all JSONL scenes from a split directory."""
    scenes = []
    for scenario_dir in Path(split_dir).iterdir():
        if not scenario_dir.is_dir():
            continue
        for scene_file in scenario_dir.glob("*.jsonl"):
            with open(scene_file) as f:
                lines = f.readlines()
                header = lines[0].strip()
                frames = [line.strip() for line in lines[1:]]
                scenes.append({
                    "scenario_type": scenario_dir.name,
                    "scene_id": scene_file.stem,
                    "header": header,
                    "frames": frames  # list of 200 frame strings
                })
    return scenes

# Load train and val splits
train_scenes = load_scenes("data_scenarios/train")
val_scenes = load_scenes("data_scenarios/val")

# Create HF dataset
dataset = DatasetDict({
    "train": Dataset.from_list(train_scenes),
    "validation": Dataset.from_list(val_scenes)
})

# Upload
dataset.push_to_hub("alexwortega/physics-scenarios", token=HF_TOKEN)
```

This will:
- Compress the dataset (saves bandwidth)
- Enable streaming (load scenes on-demand)
- Provide easier data loading via `datasets` library

## After Upload

### Model Card

Edit the model card on Hugging Face:
- https://huggingface.co/alexwortega/LFM_Physics350M

Add:
- Example outputs
- Comparison GIFs
- Training curves (if available from W&B)

### Dataset Card

Edit the dataset card:
- https://huggingface.co/datasets/alexwortega/physics-scenarios

Add:
- Scenario visualization GIFs
- Data exploration notebook link
- Statistics table

## Verify Upload

### Test Model Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2-350M",
    torch_dtype="bfloat16",
    device_map="auto"
)

model = PeftModel.from_pretrained(base, "alexwortega/LFM_Physics350M")
tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")

# Test generation
prompt = "Scene: 2 objects..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### Test Dataset Loading

```python
from datasets import load_dataset

dataset = load_dataset("alexwortega/physics-scenarios")
print(dataset)

# Load single example
example = dataset["train"][0]
print(example["header"])
print(example["frames"][0])
```

## Troubleshooting

### Large File Upload Fails

For files >5GB, use Git LFS:

```bash
git lfs install
git lfs track "*.safetensors"
git lfs track "*.bin"
git add .gitattributes
git add adapter_model.safetensors
git commit -m "Add model"
git push
```

### Dataset Upload Too Slow

Upload splits separately or use streaming dataset format.

### Authentication Errors

Make sure your token has write permissions:
```bash
huggingface-cli login
```
