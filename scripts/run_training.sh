#!/bin/bash
# Training script for GPT physics model

export PYTHONUNBUFFERED=1
cd /home/alexw

echo "Starting training at $(date)"
echo "=========================="

# Use max-examples-per-stage to limit data loading for faster iteration
# 1000 examples per stage = ~100 scenes = ~1-2 min per stage
python -u scripts/train_scratch.py \
  --data-dir data/train \
  --val-dir data/val \
  --model-size small \
  --epochs 2 \
  --batch-size 32 \
  --log-interval 10 \
  --eval-interval 100 \
  --save-interval 500 \
  --wandb-mode disabled \
  --curriculum-stages 5 \
  --max-examples-per-stage 5000 \
  --output-dir checkpoints/gpt-physics 2>&1 | tee logs/train_scratch.log

echo "=========================="
echo "Training finished at $(date)"
