#!/bin/bash
set -e
set -o pipefail

# Plucker-only single-segment evaluation runner.
# It iterates over the dataset from paths.env, but each episode only runs one segment.

# link to correct cudnn and cuda
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# source paths.env, get BASE_FOLDER and OUTPUT_ROOT
source "$SCRIPT_DIR/paths.env"

# CKPT=MODELS/evoworld_curve_unity
# CKPT=/data1/songcx/huggingface_cache/hub/models--CometsFeiyu--Evoworld_Unity_Curve_Path/snapshots/d6250ea37f38341f49dfe1009402e3684c2efc93/
CKPT=/home/user/songcx/code/EvoWorld/evo_checkpoints/unity_curve-OnlyPlucker-deepspeed_o2-lr-1e-5-step-30000-worldsize-4-length-25-only-position/onlyposition_checkpoint-10000_pipeline
# CKPT=/home/user/songcx/code/EvoWorld/evo_checkpoints/unity_curve-OnlyPlucker-deepspeed_o2-lr-1e-5-step-30000-worldsize-4-length-25/checkpoint-10000_pipeline

OUTPUT_ROOT="$Result_OUTPUT_ROOT"
SAVE_DIR="$OUTPUT_ROOT/$(basename "$CKPT")/eval_unity_curve_only_plucker_single_segment"
START_IDX=0 # starting episode index for data loading, change it if some episodes have been processed and saved in the save dir.
NUM_DATA_PER_GPU=200
NUM_SEGMENTS=1
CURVE_PATH=true
SKIP_COMPLETED=true
ONLY_POSITION=1 # set to 1 to only use position embedding, set to 0 to use full plucker embedding
CLIP_START_FRAME=1 # set to 1 to evaluate the first contiguous 25-frame clip.
DECODE_CHUNK_SIZE=8
WIDTH=1024
HEIGHT=512 # 576 | 512

echo "Running plucker-only single-segment evaluation pipeline..."
echo "Checkpoint: $CKPT"
echo "Base folder: $BASE_FOLDER"
echo "Save dir: $SAVE_DIR"
echo "Number of segments: $NUM_SEGMENTS"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Decode chunk size: $DECODE_CHUNK_SIZE"
echo "Skip completed: $SKIP_COMPLETED"
echo "ONLY_POSITION: $ONLY_POSITION"
echo "CLIP_START_FRAME: $CLIP_START_FRAME"

CMD="python unified_loop_consistency_only_plucker.py \
    --unet_path $CKPT \
    --svd_path $CKPT \
    --base_folder $BASE_FOLDER \
    --save_dir $SAVE_DIR \
    --num_data $NUM_DATA_PER_GPU \
    --start_idx $START_IDX \
    --num_segments $NUM_SEGMENTS \
    --num_frames 25 \
    --decode_chunk_size $DECODE_CHUNK_SIZE \
    --clip_start_frame $CLIP_START_FRAME \
    --width $WIDTH \
    --height $HEIGHT \
    --save_frames \
    --single_segment"

if [ "$CURVE_PATH" = true ]; then
    CMD="$CMD --curve_path"
fi

if [ "$SKIP_COMPLETED" = true ]; then
    CMD="$CMD --skip_completed"
fi

if [ "$ONLY_POSITION" = "1" ]; then
    CMD="$CMD --only_position"
fi

echo "Command: $CMD"
eval $CMD

echo "Plucker-only single-segment evaluation completed!"
echo "For metrics, use gt_subdir=predictions_gt and gen_subdir=predictions under: $SAVE_DIR"
