#!/bin/bash

export WANDB_PROJECT="embedding+llm"

python src/train.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --data_path ./data/msmarco-100k-clean-train.jsonl \
    --encoder_name jinaai/jina-embeddings-v2-base-en \
    --projector_type mlp2x_gelu \
    --tune_mlp_adapter \
    --bf16 \
    --output_dir ./checkpoints/pretrain-test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing \
    --dataloader_num_workers 4 \
    --report_to "wandb"
