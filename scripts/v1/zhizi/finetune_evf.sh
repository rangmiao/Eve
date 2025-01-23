#!/bin/bash
WORK_DIR=$(pwd)
export PYTHONPATH=${WORK_DIR}


moe_mode="sparse"
num_experts=2
top_k_experts=1
use_residual=True
router_aux_loss_coef=0.01

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed eve/train/train_xformers.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules gate_proj gate_proj2 down_proj up_proj wg \
    --deepspeed ./scripts/deepspeed/zero2.json \
    --model_name_or_path ./checkpoints/Eve-1.8b-finetune \
    --version v1 \
    --data_path  finetune_data/llava_v1_5_mix665k.json \
    --image_folder finetune_data \
    --image_tower vision_model/clip-vit-large-patch14-336  \
    --image_projector_type ldpnet \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --freeze_mm_mlp_adapter True \
    --fp16 True \
    --use_l1loss True \
    --hard_top1 False \
    --output_dir ./checkpoints/Eve-1.8b-finetune-evf \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
