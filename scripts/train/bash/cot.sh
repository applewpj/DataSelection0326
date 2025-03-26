DATASET=medqa
MODEL_PATH=/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B 
PEFT_PATH=/mnt/hwfile/medai/liaoyusheng/projects/LLM-REASONING/DataSeletion/checkpoints/medqa_5op-llama3-8b-sft
MODEL_NAME=medqa_5op-llama3-8b-sft

DATA_PATH=./dataset/test/${DATASET}.jsonl
OUTPUT_PATH=./results/${DATASET}/${MODEL_NAME}

sbatch scripts/train/slurm/cot.sh $MODEL_PATH $DATA_PATH $OUTPUT_PATH $PEFT_PATH