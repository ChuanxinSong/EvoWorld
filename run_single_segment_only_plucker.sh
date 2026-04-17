#!/bin/bash

# Plucker-only single-segment runner
# Inference matches no-memory training: first frame + plucker only

export CUDA_VISIBLE_DEVICES=2

# link to correct cudnn and cuda
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

WIDTH=1024
HEIGHT=512

# CKPT=MODELS/evoworld_curve_unity
# CKPT=/data1/songcx/huggingface_cache/hub/models--CometsFeiyu--Evoworld_Unity_Curve_Path/snapshots/d6250ea37f38341f49dfe1009402e3684c2efc93
CKPT=evo_checkpoints/unity_curve-OnlyPluckerNoMemory-deepspeed_o2-lr-1e-5-step-30000-worldsize-4-length-25

ONLY_POSITION=0 # set to 1 to only use position embedding, set to 0 to use full plucker embedding
CLIP_START_FRAME=1 # set to 1 to start from the first frame.

BASE_FOLDER=example/case_000
OUTPUT_ROOT=output_results
SAVE_DIR=$OUTPUT_ROOT/$(basename $CKPT)/plucker_only_single_demo
START_IDX=0
NUM_DATA_PER_GPU=1
NUM_SEGMENTS=3
CURVE_PATH=true
DECODE_CHUNK_SIZE=8

echo "Running plucker-only single-segment pipeline..."
echo "Mode: empty_with_traj + first frame + plucker embedding only"
echo "Checkpoint: $CKPT"
echo "Base folder: $BASE_FOLDER"
echo "Save dir: $SAVE_DIR"
echo "Number of segments: $NUM_SEGMENTS"
echo "Decode chunk size: $DECODE_CHUNK_SIZE"
echo "ONLY_POSITION: $ONLY_POSITION"
echo "CLIP_START_FRAME: $CLIP_START_FRAME"

CMD="python unified_loop_consistency_only_plucker.py \
    --unet_path $CKPT \
    --svd_path $CKPT \
    --width $WIDTH \
    --height $HEIGHT \
    --base_folder $BASE_FOLDER \
    --save_dir $SAVE_DIR \
    --num_data $NUM_DATA_PER_GPU \
    --start_idx $START_IDX \
    --num_segments $NUM_SEGMENTS \
    --num_frames 25 \
    --decode_chunk_size $DECODE_CHUNK_SIZE \
    --clip_start_frame $CLIP_START_FRAME \
    --save_frames \
    --single_segment"

if [ "$CURVE_PATH" = true ]; then
    CMD="$CMD --curve_path"
fi

if [ "$ONLY_POSITION" = "1" ]; then
    CMD="$CMD --only_position"
fi

echo "Command: $CMD"
eval $CMD

echo "Plucker-only single-segment pipeline completed!"
