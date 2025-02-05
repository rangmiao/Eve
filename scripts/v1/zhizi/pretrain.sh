#!/bin/bash
MODEL_PATH=/path/your/LLM/Model
cd Eve
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed moellava/train/train_xformers.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version plain \
    --data_path pretrain_data/blip_laion_cc_sbu_558k.json \
    --image_folder pretrain_data/images \
    --image_tower vision_model/clip-vit-large-patch14-336 \
    --image_projector_type ldpnet \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --local_debug False \
    --output_dir ./checkpoints/Eve-1.8b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"


