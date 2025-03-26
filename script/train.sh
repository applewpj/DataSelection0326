
conda activate data
cd /mnt/petrelfs/wangpingjie/DataSelection/MY

# DS_LIST=("ultra");NUM_LIST=(10000)

DS_LIST=("fingpt_sentiment" "general" );NUM_LIST=(1000 49000)
# DS_LIST=("pqaa");NUM_LIST=(10000)
DATASET=${DS_LIST[0]}
TASK=$(paste <(printf "%s\n" "${DS_LIST[@]}") <(printf "%s\n" "${NUM_LIST[@]}") | awk '{print $1"_"$2}' | paste -sd '+')



TRAIN_FILES=./dataset/converted_dataset/${DATASET}/${TASK}.jsonl
# TRAIN_FILES=./dataset/merged_dataset/${DATASET}/${TASK}.jsonl

# MODEL_PATH=/mnt/hwfile/medai/wangpingjie/models/llama3-8b
MODEL_NAME=llama3-8b
MODEL_PATH=/mnt/hwfile/medai/wangpingjie/models/${MODEL_NAME}
JOB_NAME=${MODEL_NAME}_${TASK}


export WANDB_PROJECT=DataSelection

OUTPUT_DIR=/mnt/hwfile/medai/wangpingjie/DataSelection/sft_output/${JOB_NAME}
if [[ ! -d $OUTPUT_DIR ]]; then
    mkdir -p $OUTPUT_DIR
fi

# torchrun --nproc_per_node 1 --nnodes 1 --rdzv-id=$ID --rdzv_backend c10d  \
ID=$RANDOM
srun --partition medai_llm  --cpus-per-task=8 --gres=gpu:4  --quotatype=auto  --output=${OUTPUT_DIR}/train.log --exclude=SH-IDC1-10-140-0-171 \
    deepspeed --master_port=$RANDOM \
    train/train.py --model_name_or_path $MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --train_files ${TRAIN_FILES} \
        --do_train True \
        --max_seq_length 2048 \
        --use_fast_tokenizer True \
        --lr_scheduler_type linear \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --eval_strategy no \
        --logging_steps 1 \
        --num_train_epochs 4 \
        --bf16 True \
        --tf32 False \
        --fp16 False \
        --overwrite_output_dir False \
        --report_to wandb \
        --optim adamw_torch \
        --seed 0 \
        --percentage 1.0 \
        --save_strategy epoch \
        --lora True \
        --lora_r 32 \
        --lora_alpha 64 \
        --lora_dropout 0.1 \
        --lora_target_modules q_proj k_proj v_proj o_proj up_proj gate_proj down_proj \
        --learning_rate 2e-05 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 32 \
        --save_total_limit 1 \
        --deepspeed ./deepspeed/zero2.json


