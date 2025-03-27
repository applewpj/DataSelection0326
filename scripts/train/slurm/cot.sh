#!/bin/bash
#SBATCH -J eval
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1  
#SBATCH --time=10-00:00:00
#SBATCH --mem-per-cpu=5G

MODEL_PATH="$1"
DATA_PATH="$2"
OUTPUT_PATH="$3"
PEFT_PATH="$4"


if [ -z "$PEFT_PATH" ]; then
    srun --jobid $SLURM_JOBID python eval_cot/eval_cot.py \
        --model_name_or_path ${MODEL_PATH} \
        --data_path ${DATA_PATH}  \
        --output_path ${OUTPUT_PATH} \
        --batch_size 4 \
        --max_new_tokens 2048 \
        --resume

else
    srun --jobid $SLURM_JOBID python eval_cot/eval_cot.py \
        --model_name_or_path ${MODEL_PATH} \
        --peft_path ${PEFT_PATH} \
        --data_path ${DATA_PATH} \
        --output_path ${OUTPUT_PATH} \
        --batch_size 4 \
        --max_new_tokens 2048 \
        --resume

fi