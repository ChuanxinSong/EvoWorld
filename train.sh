#!/bin/bash
# nvidia-smi
# unset HF_ENDPOINT
# unset HUGGINGFACE_HUB_ENDPOINT
# unset HF_HUB_ENDPOINT
# export HF_ENDPOINT=https://huggingface.co


# 修改为你的实际路径
BASE_FOLDER="/data2/songcx/dataset/evoworld/unity_curve"
# 刚才重投影生成的文件夹名（必须匹配）
REPROJ_NAME="rendered_panorama_vggt_open3d_camera_aligned_new_code"

# GPU settings
GPU_IDS="4" # 指定你想要使用的 GPU ID，例如 "0,1,2,3"

# configuration file, you can add more config files in the config folder
CONFIG_NAME="deepspeed_o1_4gpu"

# 指定主进程端口号（用于多进程通信）
MASTER_PORT=49502

# global seed
SEED=42

# data settings
# DATASET_NAME="Curve_Loop" # Coming Soon!
DATASET_NAME="unity_curve"
WIDTH=1024
HEIGHT=576
NUM_FRAMES=10

# model & trainer settings
# PRETRAIN_MODEL="MODELS/stable-video-diffusion-img2vid-xt-1-1"
# 使用官方 HF ID，会自动寻找缓存
PRETRAIN_MODEL="stabilityai/stable-video-diffusion-img2vid-xt-1-1" 
STEP=30000
SAVE_INTERVAL=5000
LR="1e-5"
LR_WARMUP_STEP=500
LR_SCHEDULER="cosine"
PRECISION="fp16"
# PRECISION="bf16"
VALIDATION_STEP=500
NUM_VALIDATION_IMAGES=3
RESUME_FROM="latest"
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
BATCH_SIZE_PER_GPU=1
GRAD_ACCUM_STEP=16
# GPUS_PER_NODE=$(nvidia-smi -L | wc -l) # 注释掉这行，防止覆盖你自定义的 GPU 数量
GPUS_PER_NODE=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
WORLD_SIZE=$((GPUS_PER_NODE * GRAD_ACCUM_STEP * BATCH_SIZE_PER_GPU))

# 结果保存路径
OUTPUT_ROOT="/data3/songcx/results/evoworld/checkpoints"
mkdir -p $OUTPUT_ROOT

# [Optional] use your own wandb with arg: --report_to wandb
# export WANDB_RUN_NAME="data-${DATASET_NAME}-lr-${LR}-step-${STEP}-bs-${BATCH_SIZE_PER_GPU}x${GPUS_PER_NODE}x${GRAD_ACCUM_STEP}-${CURRENT_TIME}"
# export WANDB_API_KEY=''
# export WANDB_ENTITY=''
# export WANDB_PROJECT='evoworld'
# echo "Runing will be logged to WANDB project: $WANDB_PROJECT, entity: $WANDB_ENTITY, run name: $WANDB_RUN_NAME"

# 在训练开始前，将脚本中定义的所有超参数和配置信息打印到终端上，方便检查和记录
for var in CONFIG_NAME SEED DATASET_NAME WIDTH HEIGHT NUM_FRAMES PRETRAIN_MODEL STEP SAVE_INTERVAL GRAD_ACCUM_STEP LR LR_WARMUP_STEP LR_SCHEDULER WORLD_SIZE PRECISION VALIDATION_STEP NUM_VALIDATION_IMAGES RESUME_FROM; do
    echo "$var: ${!var}"
done

echo "Current Env: $(conda info --envs | grep '*' | awk '{print $1}')"

accelerate launch --config_file="config/${CONFIG_NAME}.yaml" \
    --num_processes=$GPUS_PER_NODE \
    --gpu_ids=$GPU_IDS \
    --main_process_port=$MASTER_PORT \
    evoworld/trainer/train_evoworld.py \
    --base_folder=$BASE_FOLDER \
    --reprojection_name=$REPROJ_NAME \
    --pretrained_model_name_or_path=$PRETRAIN_MODEL \
    --num_frames=$NUM_FRAMES \
    --width=$WIDTH \
    --height=$HEIGHT \
    --output_dir="$OUTPUT_ROOT/$DATASET_NAME-$CONFIG_NAME-lr-$LR-step-$STEP-worldsize-$WORLD_SIZE-length-$NUM_FRAMES" \
    --logging_dir="$OUTPUT_ROOT/$DATASET_NAME-$CONFIG_NAME-lr-$LR-step-$STEP-worldsize-$WORLD_SIZE-length-$NUM_FRAMES/logs" \
    --per_gpu_batch_size=$BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps=$GRAD_ACCUM_STEP \
    --gradient_checkpointing \
    --max_train_steps=$STEP \
    --checkpointing_steps=$SAVE_INTERVAL \
    --checkpoints_total_limit=4 \
    --learning_rate=$LR \
    --lr_warmup_steps=$LR_WARMUP_STEP \
    --lr_scheduler=$LR_SCHEDULER \
    --scale_lr \
    --seed=$SEED \
    --mixed_precision=$PRECISION \
    --validation_steps=$VALIDATION_STEP \
    --num_validation_images=$NUM_VALIDATION_IMAGES \
    --add_plucker \
    --resume_from_checkpoint=$RESUME_FROM
