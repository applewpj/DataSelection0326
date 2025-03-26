


# warm up training
cd /mnt/petrelfs/wangpingjie/DataSelection/LESS
DATA_DIR=/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/converted_dataset/
MODEL_PATH=/mnt/hwfile/medai/wangpingjie/models/llama3-8b
PERCENTAGE=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
DATA_SEED=0
JOB_NAME=llama3-8b-p${PERCENTAGE}-lora-seed${DATA_SEED}_2
./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"



# get lora gradients for training data
cd /mnt/petrelfs/wangpingjie/DataSelection/LESS
CKPT=516
TRAINING_DATA_NAME=general
TRAINING_DATA_FILE=/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/converted_dataset/general/general_full.jsonl # when changing data name, change the data path accordingly
GRADIENT_TYPE="adam"
MODEL_PATH=/mnt/hwfile/medai/wangpingjie/DataSelection/less_output/llama3-8b-p0.05-lora-seed0/checkpoint-${CKPT}
OUTPUT_PATH=/mnt/hwfile/medai/wangpingjie/DataSelection/less_output/grads/llama3-8b-p0.05-lora-seed0/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
DIMS="8192"
./less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"

# get lora gradients for validation data
cd /mnt/petrelfs/wangpingjie/DataSelection/LESS
CKPT=516
TASK=fingpt_sentiment
MODEL_PATH=/mnt/hwfile/medai/wangpingjie/DataSelection/less_output/llama3-8b-p0.05-lora-seed0/checkpoint-${CKPT}
OUTPUT_PATH=/mnt/hwfile/medai/wangpingjie/DataSelection/less_output/grads/llama3-8b-p0.05-lora-seed0/${TASK}-ckpt${CKPT}-sgd # for validation data, we always use sgd
# DATA_DIR=../data
TRAIN_FILE=/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/converted_dataset/${TASK}/${TASK}_1000.jsonl
DIMS="8192" # We use 8192 as our default projection dimension 
./less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$TRAIN_FILE" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"


# select data
cd /mnt/petrelfs/wangpingjie/DataSelection/LESS
DIM=8192 # decide which dimension to use
GRADIENT_PATH=/mnt/hwfile/medai/wangpingjie/DataSelection/less_output/grads/llama3-8b-p0.05-lora-seed0/{}-ckpt{}-adam/dim${DIM}
TRAIN_FILE_NAMES="general"
# CKPTS="129 259 388 516" # checkpoing index
# CHECKPOINT_WEIGHTS="1.67725e-05 1.288e-5 7.68e-6 2.54e-6" # average lr of the epoch
CKPTS="516" # checkpoing index
CHECKPOINT_WEIGHTS="2.54e-6" # average lr of the epoch
VALIDATION_GRADIENT_PATH=/mnt/hwfile/medai/wangpingjie/DataSelection/less_output/grads/llama3-8b-p0.05-lora-seed0/{}-ckpt{}-sgd/dim${DIM}
TARGET_TASK_NAMES="lawinstruct_en" # fingpt_sentiment_1000 lawinstruct_en
SELECTED_DATA_OUTPUT_PATH="/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/selected_data/less"
./less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"

# write selected samples
TARGET_TASK_NAMES="lawinstruct_en" # fingpt_sentiment_1000 lawinstruct_en
TRAIN_FILE_NAMES=/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/selected_data/less/${TARGET_TASK_NAMES}/general_influence_score.pt
TRAIN_FILES=/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/converted_dataset/general/general_full.jsonl
MAX_SAMPLES=1000
SELECTED_DATA_OUTPUT_PATH="/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/selected_data/less/${TARGET_TASK_NAMES}/general(less)_${MAX_SAMPLES}.jsonl"
srun -p medai_llm --cpus-per-task=8 --time=4-00:00:00 \
    python -u less/data_selection/write_selected_data.py \
        --target_task_names ${TARGET_TASK_NAMES} \
        --train_file_names ${TRAIN_FILE_NAMES} \
        --train_files $TRAIN_FILES \
        --output_path $SELECTED_DATA_OUTPUT_PATH \
        --max_samples $MAX_SAMPLES
        # --percentage 0.05

# train with selected data
cd /mnt/petrelfs/wangpingjie/DataSelection/LESS
TARGET_TASK_NAME="pqaa"
PERCENTAGE=0.05
TRAIN_FILES=../selected_data/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
MODEL_PATH=meta-llama/Llama-2-7b-hf
JOB_NAME=llama2-7b-less-p${PERCENTAGE}-lora
./less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 