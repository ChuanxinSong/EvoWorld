#!/bin/bash
set -e

# Unified Loop Consistency Pipeline Runner
# This script runs the unified pipeline that combines generation and reconstruction

# 一张卡大概需要36小时

# link to correct cudnn and cuda
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export CUDA_VISIBLE_DEVICES=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# source paths.env, get BASE_FOLDER and OUTPUT_ROOT
source "$SCRIPT_DIR/paths.env"

# CKPT=MODELS/evoworld_curve_unity
CKPT=/data3/songcx/huggingface_cache/hub/models--CometsFeiyu--Evoworld_Unity_Curve_Path/snapshots/d6250ea37f38341f49dfe1009402e3684c2efc93/
# CKPT=/home/user/songcx/code/EvoWorld/evo_checkpoints/unity_curve-deepspeed_o2-lr-1e-5-step-30000-worldsize-4-length-25-H-576
# CKPT=/home/user/songcx/code/EvoWorld/evo_checkpoints/unity_curve-deepspeed_o2-lr-1e-5-step-30000-worldsize-4-length-25-H-512

OUTPUT_ROOT="$Result_OUTPUT_ROOT"
SAVE_DIR="$OUTPUT_ROOT/$(basename "$CKPT")/eval_unity_curve"
START_IDX=0 # starting episode index for data loading, change it if some episodes have been processed and saved in the save dir. 
NUM_DATA_PER_GPU=200
NUM_SEGMENTS=3
CURVE_PATH=true
SKIP_COMPLETED=true
WIDTH=1024
HEIGHT=576 # 576 | 512

echo "Running unified loop consistency pipeline..."
echo "Checkpoint: $CKPT"
echo "Base folder: $BASE_FOLDER"
echo "Save dir: $SAVE_DIR"
echo "Number of segments: $NUM_SEGMENTS"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Skip completed: $SKIP_COMPLETED"

CMD="python unified_loop_consistency.py \
    --unet_path $CKPT \
    --svd_path $CKPT \
    --base_folder $BASE_FOLDER \
    --save_dir $SAVE_DIR \
    --num_data $NUM_DATA_PER_GPU \
    --start_idx $START_IDX \
    --num_segments $NUM_SEGMENTS \
    --num_frames 25 \
    --clip_start_frame 1\
    --width $WIDTH \
    --height $HEIGHT \
    --save_frames"

if [ "$CURVE_PATH" = true ]; then
    CMD="$CMD --curve_path"
fi

if [ "$SKIP_COMPLETED" = true ]; then
    CMD="$CMD --skip_completed"
fi

echo "Command: $CMD"
eval $CMD

echo "Unified pipeline completed!"
