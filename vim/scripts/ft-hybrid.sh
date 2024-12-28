#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python hybrid_main.py \
    --model hybrid_mamba_efficientnet \
    --batch-size 32 \
    --lr 5e-6 \
    --min-lr 1e-5 \
    --warmup-lr 1e-5 \
    --drop-path 0.0 \
    --weight-decay 1e-8 \
    --num_workers 4 \
    --split-data-for-val True \
    --data-set FLAME \
    --is-binary True \
    --data-path /home/T2410196/VisionMamba/flame-classification/train \
    --output_dir ./output/hybrid_mamba_efficientnet \
    --epochs 50 \
    --finetune /home/T2410196/VisionMamba/Vim-tiny-midclstok/vim_tiny_73p1.pth \
    --no_amp