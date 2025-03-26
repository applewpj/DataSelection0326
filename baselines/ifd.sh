conda activate data


# calculate scores for pre-experienced data
cd /mnt/petrelfs/wangpingjie/DataSelection/Cherry_LLM
DATA_PATH=/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/converted_dataset/general/general_49000.jsonl
# DATA_PATH=/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/source_dataset/gpt4_alpaca/alpaca_gpt4_data.json
SAVE_PATH=/mnt/hwfile/medai/wangpingjie/DataSelection/ifd_output/pre-experienced/general_pre.pt 
MODEL_PATH=/mnt/hwfile/medai/wangpingjie/models/llama3-8b
srun -p medai_llm --cpus-per-task=8 --gres=gpu:1 --quotatype=auto --time=4-00:00:00 \
    python -u cherry_seletion/data_analysis.py \
        --data_path $DATA_PATH \
        --save_path $SAVE_PATH\
        --model_name_or_path $MODEL_PATH \
        --max_length 512 \
        --prompt alpaca \
        --mod pre



# select pre-experienced data
cd /mnt/petrelfs/wangpingjie/DataSelection/Cherry_LLM
PRE_PT_PATH=/mnt/hwfile/medai/wangpingjie/DataSelection/ifd_output/pre-experienced/general_pre.pt
DATA_PATH=/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/converted_dataset/general/general_49000.jsonl
SAVE_PATH=/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/selected_data/ifd/pre/general.jsonl
export OPENBLAS_NUM_THREADS=8
srun -p medai_llm --cpus-per-task=24 --time=4-00:00:00 --gres=gpu:1 --quotatype=auto \
    python -u cherry_seletion/data_by_cluster.py \
        --pt_data_path $PRE_PT_PATH \
        --json_data_path $DATA_PATH \
        --json_save_path $SAVE_PATH \
        --sample_num 10 \
        --kmeans_num_clusters 100 \
        --low_th 25 \
        --up_th 75



# train pre-experienced model
conda activate data
cd /mnt/petrelfs/wangpingjie/DataSelection/MY
# TRAIN_FILES=/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/selected_data/ifd/pre/general.jsonl
TRAIN_FILES=/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/selected_data/ifd/pre/general.jsonl
MODEL_PATH=/mnt/hwfile/medai/wangpingjie/models/llama3-8b
JOB_NAME=llama3_8b_pre
export WANDB_PROJECT=DataSelection
OUTPUT_DIR=/mnt/hwfile/medai/wangpingjie/DataSelection/ifd_output/${JOB_NAME}
if [[ ! -d $OUTPUT_DIR ]]; then
    mkdir -p $OUTPUT_DIR
fi
ID=$RANDOM
srun --partition medai_llm  --cpus-per-task=8 --gres=gpu:4  --quotatype=auto  --output=${OUTPUT_DIR}/train.log \
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
        --learning_rate 2e-05 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --save_total_limit 1 \
        --deepspeed ./deepspeed/zero2.json 
        # --lora True \
        # --lora_r 32 \
        # --lora_alpha 64 \
        # --lora_dropout 0.1 \
        # --lora_target_modules q_proj k_proj v_proj o_proj up_proj gate_proj down_proj \


# select cherry data
cd /mnt/petrelfs/wangpingjie/DataSelection/Cherry_LLM
DATA_PATH=/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/converted_dataset/general/general_full.jsonl
SAVE_PATH=/mnt/hwfile/medai/wangpingjie/DataSelection/ifd_output/analysis/general.pt
MODEL_PATH=/mnt/hwfile/medai/wangpingjie/DataSelection/ifd_output/llama3_8b_pre/checkpoint-124
srun -p medai_llm --cpus-per-task=24 --time=4-00:00:00 --gres=gpu:1 --quotatype=auto \
    python -u cherry_seletion/data_analysis.py \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH \
    --model_name_or_path $MODEL_PATH \
    --max_length 512 \
    --prompt alpaca \
    --mod cherry


cd /mnt/petrelfs/wangpingjie/DataSelection/Cherry_LLM
TASK=pqaa
NUM_SAMPLES=9000
PT_PATH=/mnt/hwfile/medai/wangpingjie/DataSelection/ifd_output/analysis/general.pt
DATA_PATH=/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/converted_dataset/general/general_full.jsonl
SAVE_PATH="/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/selected_data/ifd/${TASK}/general(ifd)_${NUM_SAMPLES}.jsonl"
TOKENIZER_PATH=/mnt/hwfile/medai/wangpingjie/models/llama3-8b
srun -p medai_llm --cpus-per-task=8 --time=4-00:00:00 \
    python -u cherry_seletion/data_by_IFD.py \
        --model_name_or_path $TOKENIZER_PATH \
        --pt_data_path $PT_PATH \
        --json_data_path $DATA_PATH \
        --json_save_path $SAVE_PATH \
        --max_length 512 \
        --sample_number $NUM_SAMPLES \
        --prompt alpaca





