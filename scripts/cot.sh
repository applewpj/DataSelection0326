DATASET=pubmedqa
MODEL_PATH=/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B 
PEFT_PATH=/mnt/hwfile/medai/liaoyusheng/projects/LLM-REASONING/DataSeletion/checkpoints/medqa_5op-llama3-8b-sft-full
MODEL_NAME=medqa_5op-llama3-8b-sft-full

srun python -m pdb eval_cot/eval_cot.py \
    --model_name_or_path ${MODEL_PATH} \
    --peft_path ${PEFT_PATH} \
    --data_path ./dataset/test/${DATASET}.jsonl \
    --output_path ./results/${DATASET}/${MODEL_NAME} \
    --batch_size 4 \
    --max_new_tokens 2048 \
    --resume