#!/bin/bash
# QLoRA 微调 v3 —— 核心修复：解冻 audio_tower 与 aligner，让模型真正"听"音频
# v2 失败根因: freeze_vit=True + freeze_aligner=True (ms-swift 默认)
#   → thinker.audio_tower / thinker.audio_tower.proj 全冻结
#   → 模型没在听音频，靠文本模板套答案
BASE=/gemini/code/lightfour-train
OUT=$BASE/checkpoints/omni3b-edm-v3
mkdir -p $BASE/logs

nohup swift sft \
  --model /gemini/pretrain \
  --tuner_type lora --tuner_backend peft \
  --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --freeze_vit false --freeze_aligner false \
  --quant_method bnb --quant_bits 4 \
  --dataset $BASE/data/sft/train_v3.jsonl \
  --val_dataset $BASE/data/sft/eval_v3.jsonl \
  --max_length 8192 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --warmup_ratio 0.05 --lr_scheduler_type cosine \
  --bf16 true --gradient_checkpointing true \
  --lazy_tokenize true --dataset_num_proc 4 --dataloader_num_workers 2 \
  --save_strategy steps --save_steps 300 --save_total_limit 12 \
  --eval_strategy steps --eval_steps 300 \
  --logging_steps 5 --report_to tensorboard \
  --output_dir $OUT \
  --seed 42 > $BASE/logs/train_v3.log 2>&1 &

echo "TRAIN_PID=$!"
echo $! > $BASE/logs/train_v3.pid
