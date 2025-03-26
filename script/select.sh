

conda activate data
cd /mnt/petrelfs/wangpingjie/DataSelection/MY

srun --partition medai_llm \
    python selection/random_select.py \
        --converted_data_dir ./dataset/converted_dataset \
        --selected_data_dir ./dataset/selected_dataset \
        --dataset dolly \
        --p 0.05