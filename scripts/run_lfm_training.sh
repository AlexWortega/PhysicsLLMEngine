#!/bin/bash
# LFM2-350M fine-tuning on diverse physics scenarios dataset
# Uses Unsloth + LoRA in physics_venv

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
cd /home/alexw

echo "=============================================="
echo "LFM2-350M Fine-tuning on Physics Scenarios"
echo "=============================================="
echo "Started at $(date)"
echo "Data: data_scenarios/train (900K scenes, 24 scenario types)"
echo "Val:  data_scenarios/val (100K scenes, 30 scenario types)"
echo ""

/home/alexw/physics_venv/bin/python3 -u scripts/train_finetune.py \
  --data-dir /home/alexw/data_scenarios/train \
  --output-dir /home/alexw/checkpoints/lfm2-scenarios \
  --epochs-per-stage 1 \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 2e-4 \
  --max-seq-length 8192 \
  --curriculum-stages 5 \
  --max-examples-per-stage 50000 \
  --max-context-frames 200 \
  --complexity-metric difficulty \
  --lora-r 32 \
  --lora-alpha 64 \
  --wandb-project physics-llm \
  --wandb-offline 2>&1 | tee logs/lfm2_scenarios_train.log

echo "=============================================="
echo "Training finished at $(date)"
