#!/bin/bash

set -e

export WANDB_PROJECT="embedding+llm"

deepspeed --include="localhost:0,1,2,3" src/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_type rank_lm \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --data_path ./data/rankgpt-100k-clean-train.jsonl \
    --encoder_name jinaai/jina-embeddings-v2-base-en \
    --pretrain_mlp_adapter ./checkpoints/vicuna.jina.wiki1m.pretrain/projector.bin \
    --projector_type mlp2x_gelu \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --output_dir ./checkpoints/vicuna.jina.wiki1m.pretrain.rankgpt100k.finetune.weighted_ranking_loss.2e-5 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb
