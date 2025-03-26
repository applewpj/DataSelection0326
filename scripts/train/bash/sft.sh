TRAINING_DATA=medqa_5op

DATA_PATH=./dataset/train/${TRAINING_DATA}.json
MODEL_PATH=/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B 
model_name=llama3-8b

ckpt_name=${TRAINING_DATA}-${model_name}-sft-full
OUTPUT_PATH=../checkpoints/${ckpt_name}

LOG_PATH=./logs/${TRAINING_DATA}

mkdir -p $LOG_PATH

sbatch -o $LOG_PATH/${ckpt_name}.log ./scripts/train/slurm/sft.sh $DATA_PATH $MODEL_PATH $OUTPUT_PATH ${DATA_USAGE} "--learning_rate 2e-5 --num_train_epochs 3 --per_device_train_batch_size 4 --gradient_accumulation_steps 8"



# bash scripts/eval/eval_models_per_dataset_med.sh None ${TRAINING_DATA} ${model_name} greedy 1 1 



