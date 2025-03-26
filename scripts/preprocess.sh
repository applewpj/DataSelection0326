

conda activate data
cd /mnt/petrelfs/wangpingjie/DataSelection/MY


# ./dataset/source_dataset
srun --partition medai_llm --time=4-00:00:00 \
    python ./data_processing/convert_dataset.py \
        --raw_data_dir ./dataset/source_dataset \
        --output_dir ./dataset/converted_dataset \
        --dataset wizardlm
        # --num_examples 1000


# merge
srun --partition medai_llm --time=4-00:00:00 \
    python ./data_processing/merge_dataset.py \
        --dataset_dir ./dataset/converted_dataset \
        --output_dir ./dataset/converted_dataset \
        --to_merge_dataset lawinstruct_en \
        --to_merge_num full \
        --select_num 50000
        # --output_name general



