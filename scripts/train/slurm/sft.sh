#!/bin/bash
#SBATCH -J sft
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1  
#SBATCH --time=10-00:00:00
#SBATCH --mem-per-cpu=5G

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun -N1 -n1 -w "$head_node" hostname --ip-address)

GPUS_PER_NODE=4
NNODES=$SLURM_NNODES


echo Node IP: $head_node_ip nodes_array: $nodes_array
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号
export LOGLEVEL=INFO
# export NCCL_SOCKET_IFNAME="eth0"
MASTER_PORT=$((RANDOM % 1001 + 20000))

export NCCL_DEBUG=ERROR
export NCCL_SOCKET_IFNAME=eth0        

DATA_PATH="$1"
# DATA_PATH=/mnt/petrelfs/jiangshuyang.p/datasets/medical_train/mix16_500_data_2.json
MODEL_PATH="$2"
# MODEL_PATH=/mnt/petrelfs/jiangshuyang.p//checkpoints/llama318b_mcts_vllm_mix16_500_data_all_trial5/sft_1-llama3.1-8b-r16a32-1epoch-SFT-full-ITER1
OUTPUT_PATH="$3"
# OUTPUT_PATH=/mnt/petrelfs/jiangshuyang.p/checkpoints/grpo_test_meds3
other_params="$4"
# --learning_rate --score_lr --learn_by_stage 

argv=()
read -ra argv <<< "$other_params"
# echo ${argv[@]}

if [[ "$other_params" != *"--learning_rate"* ]]; then 
    argv+=("--learning_rate" "1e-6")
fi

if [[ "$other_params" != *"--num_train_epochs"* ]]; then 
    argv+=("--num_train_epochs" "1")
fi

if [[ "$other_params" != *"--per_device_train_batch_size"* ]]; then 
    argv+=("--per_device_train_batch_size" "8")
fi

if [[ "$other_params" != *"--gradient_accumulation_steps"* ]]; then 
    argv+=("--gradient_accumulation_steps" "8")
fi

echo ${argv[@]}

# bash /mnt/petrelfs/jiangshuyang.p/add_oss.sh
CUDA_LAUNCH_BLOCKING=1
srun accelerate launch --config_file=scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes ${GPUS_PER_NODE} \
    --main_process_port 29305 \
    train.py \
    --bf16 True \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATA_PATH} \
    --output_dir ${OUTPUT_PATH} \
    --per_device_eval_batch_size 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --max_seq_length 4096 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to wandb \
    ${argv[@]}