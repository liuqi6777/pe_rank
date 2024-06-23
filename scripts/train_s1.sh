#!/bin/bash

deepspeed --include="localhost:0,1,2,3" src/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --data_path ./data/wiki2m.jsonl \
    --encoder_name jinaai/jina-embeddings-v2-base-en \
    --encoder_pooling mean \
    --projector_type mlp2x_gelu \
    --freeze_backbone \
    --tune_mlp_adapter \
    --bf16 \
    --output_dir ./checkpoints/mistral.jina.projector \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --dataloader_num_workers 4