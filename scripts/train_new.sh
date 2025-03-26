
TRAINING_DATA=pubmedqa
DATA_PATH=./pushed_dataset/${TRAINING_DATA}.jsonl
MODEL_PATH=/temp/liaoyusheng/LLMs/Qwen2.5-3B-Instruct/

MODEL_NAME=qwen2.5-3b
ckpt_name="${TRAINING_DATA}-${MODEL_NAME}-sft"
OUTPUT_PATH=/temp/liaoyusheng/checkpoints/${ckpt_name}
# OUTPUT_PATH=/mnt/petrelfs/jiangshuyang.p/checkpoints/Meta-Llama-3.1-8B-Instruct-ysl_medqa_grpo2
LOG_PATH=/temp/liaoyusheng/logs/${ckpt_name}.log


accelerate launch --config_file=script/accelerate_configs/deepspeed_zero2.yaml \
     train.py \
    --bf16 True \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATA_PATH} \
    --output_dir ${OUTPUT_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --max_seq_length 4096 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to wandb # > ${LOG_PATH}