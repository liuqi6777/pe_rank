#!/bin/bash

export WANDB_PROJECT="embedding+llm"

deepspeed --include="localhost:1,2,3,4" src/train.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --data_path ./data/msmarco-100k-clean-train.jsonl \
    --encoder_name bert-base-uncased \
    --pretrain_mlp_adapter ./checkpoints/vicuna.jina.rankgpt100k.pretrain/projector.bin \
    --projector_type mlp2x_gelu \
    --bf16 True \
    --output_dir ./checkpoints/vicuna.jina.rankgpt100k.lora \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb
