DATASET=pubmedqa
MODEL_PATH=/temp/liaoyusheng/LLMs/Qwen2.5-3B-Instruct/
MODEL_NAME=qwen2.5-3b

python -m pdb eval_cot/eval_cot.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ./pushed_dataset/${DATASET}.jsonl \
    --output_path ./results/${DATASET}/${MODEL_NAME} \
    --batch_size 4 \
    --max_new_tokens 50 \
    --resume