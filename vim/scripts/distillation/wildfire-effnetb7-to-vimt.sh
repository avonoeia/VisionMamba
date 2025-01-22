#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --finetune /home/T2410196/VisionMamba/Vim-tiny-midclstok/vim_tiny_73p1.pth \
    --batch-size 32 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --warmup-lr 1e-5 \
    --drop-path 0.0 \
    --weight-decay 1e-8 \
    --early-stopping \
    --early-stopping-patience 5 \
    --early-stopping-delta 0.0 \
    --sched plateau \
    --num_workers 4 \
    --split-data-for-val True \
    --data-set FLAME \
    --is-binary True \
    --data-path /home/T2410196/VisionMamba/flame-classification/train \
    --output_dir ./output/distillation/finetuned/efficientnetb7_to_vim_tiny_hard_a0.75 \
    --epochs 50 \
    --no_amp \
    --teacher-model efficientnet-b7 \
    --teacher-path /home/T2410196/VisionMamba/vim/output/efficientnet-b7/efficientnet-model.pth \
    --distillation-type hard \
    --distillation-alpha 0.75 \
    --distillation-tau 20 \
